#include <cstdint>
#include <memory.h>
#include <cstdio>
#include <ctime>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <string>
#include <fstream>

#define RANDOM_MULTIPLIER 0x5DEECE66DULL
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48ULL) - 1ULL)

#ifndef FLOOR_LEVEL
#define FLOOR_LEVEL 63LL
#endif

#ifndef WANTED_CACTUS_HEIGHT
#define WANTED_CACTUS_HEIGHT 10LL
#endif

#ifndef WORK_UNIT_SIZE
#define WORK_UNIT_SIZE 2048
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

    __device__ inline int32_t next_int_unknown(uint64_t *seed, int16_t bound) {
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

struct seed {
    uint64_t seed;
    int16_t posX;
    int16_t posZ;
    int32_t height;
};

__global__ __launch_bounds__(256, 2) void crack(int32_t *num_seeds, seed *seeds, const uint64_t *input_seeds) {
    int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t originalSeed = input_seeds[id];
    uint64_t seed = originalSeed;

    int8_t heightMap[1024];

#pragma unroll
    for (int i = 0; i < 1024; i++) {
        heightMap[i] = FLOOR_LEVEL;
    }

    int16_t currentHighestPos = 0;
    int16_t terrainHeight;
    int16_t initialPosX, initialPosY, initialPosZ;
    int16_t posX, posY, posZ, sPosX, sPosZ;
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
                    sPosX = posX;
                    sPosZ = posZ;
                }
            }
        }

        if (heightMap[currentHighestPos] - FLOOR_LEVEL >= WANTED_CACTUS_HEIGHT) {
            seeds[atomicAdd(num_seeds, 1)] = { originalSeed, sPosX, sPosZ, (int32_t)(heightMap[currentHighestPos] - FLOOR_LEVEL) };
            return;
        }
    }
}

struct GPU_Node {
    int* num_seeds;
    seed* seeds;
    uint64_t* input_seeds;
};

int main() {
    printf("Searching %ld total seeds...\n", END - OFFSET);

    std::vector<uint64_t> input;
    std::ifstream inputFile("out.txt");
    while (inputFile.good()) {
        int32_t height;
        uint64_t seed;
        inputFile >> height;
        inputFile >> seed;
        inputFile.ignore(1);
        input.push_back(seed);
    }
    uint64_t count = 0;
    uint64_t offset = OFFSET;
    printf("Seed count: %d\n", (int)input.size());

    FILE *out_file = fopen("seeds.txt", "w");

    GPU_Node node{};

    cudaMallocManaged(&node.num_seeds, sizeof(*node.num_seeds));
    cudaMallocManaged(&node.seeds, sizeof(seed) * WORK_UNIT_SIZE);
    cudaMallocManaged(&node.input_seeds, sizeof(uint64_t) * WORK_UNIT_SIZE);

    while (offset < input.size()) {
        printf("Offset: %lu\n", offset);
        *node.num_seeds = 0;

        for (int i = 0; i < WORK_UNIT_SIZE; i++) {
            node.input_seeds[i] = input[i + offset];
        }

        crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>> (node.num_seeds, node.seeds, node.input_seeds);
        offset += WORK_UNIT_SIZE;
        cudaDeviceSynchronize();
        for (int32_t i = 0, e = *node.num_seeds; i < e; i++) {
            fprintf(out_file, "%d %lld %d %d\n", node.seeds[i].height, (long long int)node.seeds[i].seed, node.seeds[i].posX - 8, node.seeds[i].posZ - 8);
            printf("Found seed: %lld\n", (long long int)node.seeds[i].seed);
        }
        fflush(out_file);
        count += *node.num_seeds;
    }
    fclose(out_file);

    printf("Done!\n");
    printf("But, verily, it be the nature of dreams to end.\n");

}