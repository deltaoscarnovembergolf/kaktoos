#include <cstdint>
#include <memory.h>
#include <cstdio>
#include <ctime>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>

#define RANDOM_MULTIPLIER 0x5DEECE66DULL
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48ULL) - 1ULL)

#ifndef FLOOR_LEVEL
#define FLOOR_LEVEL 63ULL
#endif

#ifndef WANTED_CACTUS_HEIGHT
#define WANTED_CACTUS_HEIGHT 8ULL
#endif

#ifndef WORK_UNIT_SIZE
#define WORK_UNIT_SIZE (1ULL << 23ULL)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256ULL
#endif

#ifndef GPU_COUNT
#define GPU_COUNT 1ULL
#endif

#ifndef OFFSET
#define OFFSET 0
#endif

#ifndef END
#define END (1ULL << 48ULL)
#endif

namespace java_random {

    // Random::next(bits)
    __device__ inline uint32_t next(uint64_t *random, int32_t bits) {
        *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
        return (uint32_t) (*random >> (48ULL - bits));
    }

    __device__ int32_t next_int_unknown(uint64_t *seed, int16_t bound) {
        if ((bound & -bound) == bound) {
            *seed = (*seed * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
            return (int32_t) ((bound * (*seed >> 17ULL)) >> 31ULL);
        }

        int32_t bits, value;
        do {
            *seed = (*seed * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
            bits = *seed >> 17ULL;
            value = bits % bound;
        } while (bits - value + (bound - 1) < 0);
        return value;
    }

    // Random::nextInt(bound)
    __device__ inline uint32_t next_int(uint64_t *random) {
        return java_random::next(random, 31) % 3;
    }

}

__global__ void crack(uint64_t seed_offset, int32_t *num_seeds, uint64_t *seeds) {
    uint64_t originalSeed = blockIdx.x * blockDim.x + threadIdx.x + seed_offset;
    uint64_t seed = originalSeed;

    int8_t heightMap[1024];

    for (auto &temp : heightMap) {
        temp = FLOOR_LEVEL;
    }

    int16_t currentHighestPos = 0;
    int16_t terrainHeight;
    int16_t initialPosX, initialPosY, initialPosZ;
    int16_t posX, posY, posZ;
    int16_t offset, posMap;

    int16_t i, a, j;

    for (i = 0; i < 10; i++) {
        // Keep, most threads finish early this way
        if (WANTED_CACTUS_HEIGHT - heightMap[currentHighestPos] + FLOOR_LEVEL > 9 * (10 - i))
            return;

        initialPosX = java_random::next(&seed, 4) + 8;
        initialPosZ = java_random::next(&seed, 4) + 8;
        terrainHeight = (heightMap[initialPosX + initialPosZ * 32] + 1) * 2;

        initialPosY = java_random::next_int_unknown(&seed, terrainHeight);

        for (a = 0; a < 10; a++) {
            posX = initialPosX + java_random::next(&seed, 3) - java_random::next(&seed, 3);
            posY = initialPosY + java_random::next(&seed, 2) - java_random::next(&seed, 2);
            posZ = initialPosZ + java_random::next(&seed, 3) - java_random::next(&seed, 3);

            posMap = posX + posZ * 32;
            // Keep
            if (posY <= heightMap[posMap] && posY >= 0)
                continue;

            offset = 1 + java_random::next_int_unknown(&seed, java_random::next_int(&seed) + 1);

            for (j = 0; j < offset; j++) {
                if ((posY + j - 1) > heightMap[posMap] || posY < 0) continue;
                if ((posY + j) <= heightMap[(posX + 1) + posZ * 32] && posY >= 0) continue;
                if ((posY + j) <= heightMap[posX + (posZ - 1) * 32] && posY >= 0) continue;
                if ((posY + j) <= heightMap[(posX - 1) + posZ * 32] && posY >= 0) continue;
                if ((posY + j) <= heightMap[posX + (posZ + 1) * 32] && posY >= 0) continue;

                heightMap[posMap]++;

                if (heightMap[currentHighestPos] < heightMap[posMap]) {
                    currentHighestPos = posMap;
                }
            }
        }

        if (heightMap[currentHighestPos] - FLOOR_LEVEL >= WANTED_CACTUS_HEIGHT) {
            seeds[atomicAdd(num_seeds, 1)] = originalSeed;
            return;
        }
    }
}

struct GPU_Node {
    int* num_seeds;
    uint64_t* seeds;
};

void setup_gpu_node(GPU_Node* node, int32_t gpu) {
    cudaSetDevice(gpu);
    cudaMallocManaged(&node->num_seeds, sizeof(*node->num_seeds));
    cudaMallocManaged(&node->seeds, 1ULL << 10ULL); // approx 1kb
}

GPU_Node nodes[GPU_COUNT];
uint64_t offset = OFFSET;
uint64_t count = 0;
std::mutex info_lock;

void gpu_manager(int32_t gpu_index) {
    std::string fileName = "kaktoos_seeds" + std::to_string(gpu_index) + ".txt";
    FILE *out_file = fopen(fileName.c_str(), "w");
    cudaSetDevice(gpu_index);
    while (offset < END) {
        *nodes[gpu_index].num_seeds = 0;
        crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>> (offset, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds);
        info_lock.lock();
        offset += WORK_UNIT_SIZE;
        info_lock.unlock();
        cudaDeviceSynchronize();
        for (int32_t i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
            fprintf(out_file, "%lld\n", (long long int)nodes[gpu_index].seeds[i]);
            printf("Found seed: %lld\n", (long long int)nodes[gpu_index].seeds[i]);
        }
        fflush(out_file);
        info_lock.lock();
        count += *nodes[gpu_index].num_seeds;
        info_lock.unlock();
    }
    fclose(out_file);
}

int main() {
    printf("Searching %ld total seeds...\n", END - OFFSET);

    std::thread threads[GPU_COUNT];

    time_t startTime = time(nullptr), currentTime;
    for(int32_t i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i], i);
        threads[i] = std::thread(gpu_manager, i);
    }

    using namespace std::chrono_literals;

    while (offset < END) {
        time(&currentTime);
        int timeElapsed = (int)(currentTime - startTime);
        double speed = (double)(offset - OFFSET) / (double)timeElapsed / 1000000.0;
        printf("Searched %lld seeds, offset: %lld found %lld matches. Time elapsed: %ds. Speed: %.2fm seeds/s. %f%%\n",
            (long long int)(offset - OFFSET),
            (long long int)offset,
            (long long int)count,
            timeElapsed,
            speed,
            (double)(offset - OFFSET) / (END - OFFSET) * 100);

        std::this_thread::sleep_for(0.5s);
    }

    for (auto &thread : threads) {
        thread.join();
    }

}
