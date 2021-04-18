#ifndef WR_FRAMEBUFFER_H
#define WR_FRAMEBUFFER_H

#include <glad/glad.h>

#include "texture.h"

struct FrameBuffer {
    FrameBuffer(uint32_t width, uint32_t height);

    inline void bind() { glBindFramebuffer(GL_FRAMEBUFFER, this->renderer_id); };
    inline void unbind() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }

    inline GLuint get_color_attachment() { return this->color_attachment; };

private:
    GLuint renderer_id;
    GLuint color_attachment;
};

#endif // WR_FRAMEBUFFER_H