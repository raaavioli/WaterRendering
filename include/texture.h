#ifndef WR_TEXTURE_H
#define WR_TEXTURE_H

#include <iostream>
#include <vector>

#include <glad/glad.h>

/** TODO: Make more generic, do not assume byte-sized data for image */

struct Image {
    Image(uint32_t width, uint32_t height, GLenum gl_format);
    ~Image();

    void set_pixel(uint32_t x, uint32_t y, GLubyte r, GLubyte g, GLubyte b);
    inline uint32_t get_width() { return this->width; }
    inline uint32_t get_height() { return this->height; }
    inline GLenum get_gl_format() { return this->gl_format; }
    inline const char* get_data() { return this->data; }

private:
    char* data;
    uint32_t width, height;
    GLenum gl_format;
};

struct Texture2D {
    Texture2D();
    Texture2D(const char* filename);
    Texture2D(Image image);

    inline GLuint get_texture_id() { return this->renderer_id; };
    inline void bind(uint32_t slot) const {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, this->renderer_id);
    };

    inline void unbind() const { glBindTexture(GL_TEXTURE_2D, 0); };

private:
    GLuint renderer_id;
};

struct TextureCubeMap {
    TextureCubeMap(const char* filename, bool folder = false);

    inline GLuint get_texture_id() { return this->renderer_id; };
    inline void bind(uint32_t slot) const {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_CUBE_MAP, this->renderer_id);
    };

    inline void unbind() const { glBindTexture(GL_TEXTURE_CUBE_MAP, 0); };

private:
    GLuint renderer_id;

    /**
     * Load cube map images from folder within assets/ directory. Folder should contain images named: 
     *      px, nx, py, ny, pz and nz.
     * Assuming JPG 
     * TODO: Look for file extension
     */
    void from_folder(const char* foldername);

    /**
     * Load cube map image from file within assets/ directory. Should contain 6 subimages. 
     * Image dimensions needs to be 4:3, so that all subimages are square.
     * Assuming JPG 
     * TODO: Look for file extension
     * TODO: Enable more than one format
     */
    void from_file(const char* filename);
};

#endif // WR_TEXTURE_H