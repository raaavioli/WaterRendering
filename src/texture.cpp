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

Texture2D::Texture2D() {
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

Texture2D::Texture2D(const char* filename) {
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

Texture2D::Texture2D(Image image) {
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

/**
 * Create a cube map texture
 * 
 * @param filenames The file names to sample cube textures from.
 *  Should be ordered in cube map texture target order:
 *      +x, -x, +y, -y, +z, -z
 */
TextureCubeMap::TextureCubeMap(std::vector<const char*> filenames) {
    glGenTextures(1, &this->renderer_id);
    glBindTexture(GL_TEXTURE_CUBE_MAP, this->renderer_id);
    int width = 0;
    int height = 0;
    int channels = 0;
    for(uint32_t i = 0; i < filenames.size(); i++)
    {
        int tmp_width = width;
        int tmp_height = height;
        int tmp_channels = channels;
        u_char *data = stbi_load(filenames[i], &width, &height, &channels, 0);
        if (i > 0 && (width != height || width != tmp_width || height != tmp_height || channels != tmp_channels)) {
            glDeleteTextures(1, &this->renderer_id);
            std::cout << "Error: Failed to load texture: " << filenames[i] << " (wrong dimensions)" << std::endl;
            break;
        }

        if (data && channels == 3) {
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
            );
        }
        else {
            std::cout << "Error: Failed to load texture: " << filenames[i] << " channels: " << channels << std::endl;
        }
        stbi_image_free(data);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); 

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    unbind(); 
}