#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define float32_t float
#define bendian __builtin_bswap64
#define BUFFER_SIZE 512

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float x, y;
} float2;

typedef struct Vertex {
    float32_t xPos;
    float32_t yPos;
    float32_t zPos;
    uint32_t normal;
    uint32_t color;
    float32_t u;
    float32_t v;
    uint32_t data;
    uint32_t tangent;
} Vertex;

typedef struct Mesh {
    uint64_t vertices_num;
    uint64_t indices_num;
    Vertex* vertices;
    uint32_t* indices;
} Mesh;

static inline int16_t float_to_int16(float v) {
    return (int16_t)(fmaxf(-1.0f, fminf(v, 1.0f)) * 32767.0f);
}

Vec3 normalize(Vec3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len == 0.0f) return (Vec3){0, 0, 0};
    return (Vec3){v.x / len, v.y / len, v.z / len};
}

Vec3 cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

uint32_t encode_octahedral(Vec3 n) {
    float invLen = 1.0f / (fabsf(n.x) + fabsf(n.y) + fabsf(n.z));
    Vec3 v = {n.x * invLen, n.y * invLen, n.z * invLen};

    float x = v.x;
    float y = v.y;

    if (v.z < 0.0f) {
        float ox = (1.0f - fabsf(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        float oy = (1.0f - fabsf(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = ox;
        y = oy;
    }

    int16_t ix = float_to_int16(x);
    int16_t iy = float_to_int16(y);

    return ((uint16_t)ix << 16) | ((uint16_t)iy & 0xFFFF);
}

int compare_vertex(Vertex* a, Vertex* b) {
    if (fabsf(a->xPos - b->xPos) > 0.001f ||
        fabsf(a->yPos - b->yPos) > 0.001f ||
        fabsf(a->zPos - b->zPos) > 0.001f)
        return -1;

    if (fabsf(a->u - b->u) > 0.001f || fabsf(a->v - b->v) > 0.001f)
        return -1;

    if (a->normal != b->normal)
        return -1;

    if (a->tangent != b->tangent)
        return -1;

    return 0;
}

Mesh* import_ply(char* path) {
    char buff[BUFFER_SIZE];
    FILE* fp = fopen(path,"rb");
    if(fp == NULL) { fprintf(stderr,"Cannot open %s\n",path); return NULL; }

    fgets(buff,BUFFER_SIZE,fp); buff[strcspn(buff, "\n")] = 0;
    if(strcmp(buff,"ply") != 0) { fprintf(stderr,"%s is not a ply file\n",path); fclose(fp); return NULL; }

    fgets(buff,BUFFER_SIZE,fp); buff[strcspn(buff, "\n")] = 0;
    if(strcmp(buff,"format binary_little_endian 1.0") != 0) {
        fprintf(stderr,"%s is not in binary_little_endian 1.0 format\n",path);
        fclose(fp);
        return NULL;
    }

    uint64_t orig_vertices_num = 0, orig_indices_num = 0;
    while (fgets(buff, BUFFER_SIZE, fp)) {
        if (strncmp(buff, "element vertex", 14) == 0) {
            sscanf(buff, "element vertex %llu", &orig_vertices_num);
        } else if (strncmp(buff, "element face", 12) == 0) {
            sscanf(buff, "element face %llu", &orig_indices_num);
        } else if (strncmp(buff, "end_header", 10) == 0) {
            break;
        }
    }

    float* raw_positions = malloc(sizeof(float) * orig_vertices_num * 8);
    fread(raw_positions, sizeof(float), orig_vertices_num * 8, fp);

    uint32_t* face_indices = malloc(sizeof(uint32_t) * orig_indices_num * 3);
    uint64_t face_count = 0;
    for(uint64_t i = 0; i < orig_indices_num; i++) {
        uint8_t count;
        fread(&count, sizeof(uint8_t), 1, fp);
        if(count != 3) {
            fprintf(stderr,"Only triangle faces are supported.\n");
            free(raw_positions); free(face_indices); return NULL;
        }
        fread(&face_indices[face_count], sizeof(uint32_t), 3, fp);
        // Fix winding
        uint32_t tmp = face_indices[face_count + 1];
        face_indices[face_count + 1] = face_indices[face_count + 2];
        face_indices[face_count + 2] = tmp;
        face_count += 3;
    }
    fclose(fp);

    Vertex* unique_verts = malloc(sizeof(Vertex) * face_count);
    uint32_t* remapped_indices = malloc(sizeof(uint32_t) * face_count);

    Vec3* tangents_temp = calloc(face_count, sizeof(Vec3));

    // Initialize vertices (normals ignored here, replaced later)
    for (uint64_t i = 0; i < face_count; i++) {
        uint32_t idx = face_indices[i];
        float* base = &raw_positions[idx * 8];
        Vertex v = {
            .xPos = -base[0] * 16,
            .yPos =  base[1] * 16,
            .zPos =  base[2] * 16,
            .u    =  base[6],
            .v    = -base[7],
            .color = 0xFFFFFFFF,
            .data = 0,
            .normal = 0,     // will be overwritten per face
            .tangent = 0
        };
        unique_verts[i] = v;
        remapped_indices[i] = i;
    }

    // Calculate per-face normals and tangents, assign to each vertex
    for (uint64_t i = 0; i < face_count; i += 3) {
        Vertex* v0 = &unique_verts[i];
        Vertex* v1 = &unique_verts[i+1];
        Vertex* v2 = &unique_verts[i+2];

        Vec3 p0 = {v0->xPos, v0->yPos, v0->zPos};
        Vec3 p1 = {v1->xPos, v1->yPos, v1->zPos};
        Vec3 p2 = {v2->xPos, v2->yPos, v2->zPos};

        float2 uv0 = {v0->u, v0->v};
        float2 uv1 = {v1->u, v1->v};
        float2 uv2 = {v2->u, v2->v};

        Vec3 edge1 = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
        Vec3 edge2 = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};

        Vec3 face_normal = normalize(cross(edge1, edge2));

        uint32_t encoded_normal =
            ((uint8_t)((face_normal.x + 1.f) * 0.5f * 255)) |
            (((uint8_t)((face_normal.y + 1.f) * 0.5f * 255)) << 8) |
            (((uint8_t)((face_normal.z + 1.f) * 0.5f * 255)) << 16);

        v0->normal = encoded_normal;
        v1->normal = encoded_normal;
        v2->normal = encoded_normal;

        float deltaU1 = uv1.x - uv0.x;
        float deltaV1 = uv1.y - uv0.y;
        float deltaU2 = uv2.x - uv0.x;
        float deltaV2 = uv2.y - uv0.y;

        float f = 1.0f / (deltaU1 * deltaV2 - deltaU2 * deltaV1);
        Vec3 tangent = {
            f * (deltaV2 * edge1.x - deltaV1 * edge2.x),
            f * (deltaV2 * edge1.y - deltaV1 * edge2.y),
            f * (deltaV2 * edge1.z - deltaV1 * edge2.z)
        };
        tangent = normalize(tangent);

        tangents_temp[i]   = tangent;
        tangents_temp[i+1] = tangent;
        tangents_temp[i+2] = tangent;
    }

    for (uint64_t i = 0; i < face_count; i++) {
        unique_verts[i].tangent = encode_octahedral(tangents_temp[i]);
    }
    free(tangents_temp);
    free(raw_positions);
    free(face_indices);

    Mesh* mesh = malloc(sizeof(Mesh));
    mesh->vertices_num = face_count;
    mesh->indices_num = face_count;
    mesh->vertices = unique_verts;
    mesh->indices = remapped_indices;
    return mesh;
}

int main(int argc, char** argv) {
    if(argc < 3) {
        fprintf(stderr,"Usage: converter <input_file> <output_file>\n");
        exit(1);
    }

    Mesh* mesh = import_ply(argv[1]);
    if(mesh == NULL) { fprintf(stderr,"Error importing model\n"); exit(1); }

    FILE* fp = fopen(argv[2],"wb");
    if(fp==NULL) { fprintf(stderr,"Cannot open output file\n"); exit(1); }

    uint64_t format_indicator = 2, dimensions_x = 1, dimensions_y = 1, dimensions_z = 1;
    uint8_t meshes_num = 1;
    uint64_t vertices_num_bendian = bendian(mesh->vertices_num);
    uint64_t indices_num_bendian = bendian(mesh->indices_num);

    fwrite(&format_indicator,sizeof(uint64_t),1,fp);
    fwrite(&dimensions_x,sizeof(uint64_t),1,fp);
    fwrite(&dimensions_y,sizeof(uint64_t),1,fp);
    fwrite(&dimensions_z,sizeof(uint64_t),1,fp);
    fwrite(&meshes_num,sizeof(uint8_t),1,fp);
    fwrite(&vertices_num_bendian,sizeof(uint64_t),1,fp);
    fwrite(&indices_num_bendian,sizeof(uint64_t),1,fp);
    fwrite(mesh->vertices,sizeof(Vertex),mesh->vertices_num,fp);
    fwrite(mesh->indices,sizeof(uint32_t),mesh->indices_num,fp);

    free(mesh->vertices);
    free(mesh->indices);
    free(mesh);
    fclose(fp);
    return 0;
}
