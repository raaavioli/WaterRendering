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

    std::string filepath = "assets/" + std::string(filename);
    unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 4);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Error: Failed to load texture: " << filepath.c_str() << std::endl;
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
 * Create a cube map texture from skybox image
 * 
 * Supports: PNG, JPEG
 */
TextureCubeMap::TextureCubeMap(const char* filename, bool folder) {
    glGenTextures(1, &this->renderer_id);
    if (folder)
        from_folder(filename);
    else
        from_file(filename);
}

/**
 * Load images from folder within assets/ directory. Folder should contain images named: 
 *      px, nx, py, ny, pz and nz.
 * Assuming JPG 
 * TODO: Look for file extension
 */
void TextureCubeMap::from_folder(const char* foldername) {
    glBindTexture(GL_TEXTURE_CUBE_MAP, this->renderer_id);
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<const char*> files = {"px.jpg", "nx.jpg", "py.jpg", "ny.jpg", "pz.jpg", "nz.jpg"};
    for(uint32_t i = 0; i < 6; i++)
    {
        int tmp_width = width;
        int tmp_height = height;
        int tmp_channels = channels;
        std::string filepath = "assets/" + std::string(foldername) + "/" + std::string(files[i]);
        u_char *data = stbi_load(filepath.c_str(), &width, &height, &channels, 4);
        if (i > 0 && (width != height || width != tmp_width || height != tmp_height || channels != tmp_channels)) {
            glDeleteTextures(1, &this->renderer_id);
            stbi_image_free(data);
            std::cout << "Error: Failed to load texture: " << filepath.c_str() << " (wrong dimensions)" << std::endl;
            break;
        }

        if (data) {
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data
            );
        }
        else {
            std::cout << "Error: Failed to load texture: " << filepath.c_str() << " channels: " << channels << std::endl;
        }
        stbi_image_free(data);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); 
    unbind(); 
}

void TextureCubeMap::from_file(const char* filename) {
    glBindTexture(GL_TEXTURE_CUBE_MAP, this->renderer_id);
    int width = 0, height = 0, channels = 0;
    int req_channels = 4;

    std::string filepath = "assets/" + std::string(filename);
    u_char *data = stbi_load(filepath.c_str(), &width, &height, &channels, req_channels);
    if (width / 4.0 != height / 3.0) {
        glDeleteTextures(1, &this->renderer_id);
        std::cout << "Error: Failed to load texture: " << filepath.c_str() << " Wrong dimensions: " << width << ", " << height << std::endl;
        stbi_image_free(data);
        return;
    }

    if (data) {
        /** Divide data into smaller sub-images. Assuming 4 channels per color.
         * Following shows expected positions of subimages:
         * XIXX
         * IIII
         * XIXX
         */
        int dim = width / 4;
        int sub_dim_x = dim * req_channels;
        u_char px[dim * sub_dim_x];
        u_char nx[dim * sub_dim_x];
        u_char py[dim * sub_dim_x];
        u_char ny[dim * sub_dim_x];
        u_char pz[dim * sub_dim_x];
        u_char nz[dim * sub_dim_x];
        std::cout << width << ", " << height << ",  " << dim << std::endl;
        for (int y = 0; y < dim; y++) {
            for (int x = 0; x < dim * req_channels; x++) {
                int sub_offset_y = y * sub_dim_x;
                int full_offset_y = width * req_channels;
                py[sub_offset_y + x] = data[0 * dim * full_offset_y + y * full_offset_y + (1 * sub_dim_x + x)];
                nx[sub_offset_y + x] = data[1 * dim * full_offset_y + y * full_offset_y + (0 * sub_dim_x + x)];
                pz[sub_offset_y + x] = data[1 * dim * full_offset_y + y * full_offset_y + (1 * sub_dim_x + x)];
                px[sub_offset_y + x] = data[1 * dim * full_offset_y + y * full_offset_y + (2 * sub_dim_x + x)];
                nz[sub_offset_y + x] = data[1 * dim * full_offset_y + y * full_offset_y + (3 * sub_dim_x + x)];
                ny[sub_offset_y + x] = data[2 * dim * full_offset_y + y * full_offset_y + (1 * sub_dim_x + x)];
            }
        }
        std::vector<u_char*>sub_images = {px, nx, py, ny, pz, nz};
        for (int i = 0; i < 6; i++) {
            // External format GL_RGBA due to stb_load with 4 requested channels.
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                0, GL_RGBA, dim, dim, 0, GL_RGBA, GL_UNSIGNED_BYTE, sub_images[i]
            );
        }
    }
    else {
        std::cout << "Error: Failed to load texture: " << filename << " channels: " << channels << std::endl;
    }
    stbi_image_free(data);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE); 
    unbind(); 
}