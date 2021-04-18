#include "texture.h"

#include <iostream>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image::Image(uint32_t w, uint32_t h, GLenum fmt) : width(w), height(h), gl_format(fmt) {
    switch (this->gl_format) {
        case GL_RGB: {
            uint32_t size = width * height * 3 * sizeof(GLubyte);
            this->data = new char[size];

            // Default black texture 
            memset(this->data, 0, size);
            break;
        }
        default: {
            std::cout << "ERROR (Image): Non supported gl_format for image (" << gl_format << ")" << std::endl;
            break;
        }
    }
}

Image::~Image() { 
    delete[] data; 
};

void Image::set_pixel(uint32_t x, uint32_t y, GLubyte r, GLubyte g, GLubyte b) {
    switch (this->gl_format) {
        case GL_RGB: {
            uint32_t pixel = y * this->width + x;
            this->data[pixel * 3 + 0] = r;
            this->data[pixel * 3 + 1] = g;
            this->data[pixel * 3 + 2] = b;
            break;
        }
        default: {
            std::cout << "ERROR (set_pixel): Non supported gl_format for image (" << gl_format << ")" << std::endl;
            break;
        }
    }
}

Texture::Texture() {
    glGenTextures(1, &this->renderer_id);
    glBindTexture(GL_TEXTURE_2D, this->renderer_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    char data[] = {(char) 255, (char) 255, (char) 255, (char) 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    unbind();
}

Texture::Texture(const char* filename) {
    glGenTextures(1, &this->renderer_id);
    glBindTexture(GL_TEXTURE_2D, this->renderer_id);
    // set the texture wrapping/filtering options (on the currently bound texture object)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load and generate the texture
    int width, height, nrChannels;
    unsigned char *data = stbi_load(filename, &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Error: Failed to load texture: " << filename << std::endl;
    }
    stbi_image_free(data);
    unbind();
}

Texture::Texture(Image image) {
    glGenTextures(1, &this->renderer_id);
    glBindTexture(GL_TEXTURE_2D, this->renderer_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Assuming GL_UNSIGNED_BYTE for data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        image.get_width(), image.get_height(), 0,
        image.get_gl_format(), GL_UNSIGNED_BYTE, image.get_data());
    glGenerateMipmap(GL_TEXTURE_2D);
    unbind();
}