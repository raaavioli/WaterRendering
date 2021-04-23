#ifndef WR_MODEL_H
#define WR_MODEL_H

#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

#include "texture.h"

struct Vertex {
  glm::vec3 position;
  glm::vec3 color;
  glm::vec3 normal;
  glm::vec2 uv;
};

struct RawModel {
    RawModel() {};
    RawModel(const std::vector<Vertex>& data, const std::vector<uint32_t>& indices, GLenum usage);

    void update_vertex_data(const std::vector<Vertex>& vertices);

    inline void bind() { glBindVertexArray(this->renderer_id); }
    inline void unbind() { glBindVertexArray(0); }
    inline void draw() { glDrawElements(GL_TRIANGLES, this->index_count, GL_UNSIGNED_INT, 0); }

private:
    GLuint renderer_id, vbo, ebo;
    GLenum gl_usage;
    uint32_t index_count;
};

struct Skybox {
    /**
     * Create a sky box from a list of image names
     * 
     * @param filenames The file names to sample cube textures from.
     *  Should be ordered in cube map texture target order:
     *      +x, -x, +y, -y, +z, -z
     */
    Skybox(const std::vector<const char*> filenames);

    /**
     * Draw a skybox, assumes appropriate shader is used
     */
    void draw();

    inline void bind_cube_map(uint32_t slot) const { cube_map.bind(slot); };
    inline void unbind_cube_map() const { cube_map.unbind(); };

private:
    TextureCubeMap cube_map;
    GLuint renderer_id, vbo, ebo;
    const uint32_t index_count = 36;
};

#endif //WR_MODEL_H