#ifndef WR_TEXTURE_H
#define WR_TEXTURE_H

#include <iostream>

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

struct Texture {
    Texture();
    Texture(const char* filename);
    Texture(Image image);

    inline GLuint get_texture_id() { return this->renderer_id; };
    inline void bind(uint32_t slot) const {
        glActiveTexture(GL_TEXTURE0 + slot);
        glBindTexture(GL_TEXTURE_2D, this->renderer_id);
    };

    inline void unbind() const { glBindTexture(GL_TEXTURE_2D, 0); };

private:
    GLuint renderer_id;
};

#endif // WR_TEXTURE_H