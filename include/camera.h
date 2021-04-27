#ifndef WR_CAMERA_H
#define WR_CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

struct Camera {
    Camera(glm::vec3 pos, 
        float r_yaw, float r_pitch, 
        float d_fovy, float aspect_ratio, float near_clip, float far_clip, 
        float rot_speed, float mov_speed
    ) : position(pos), 
        yaw(r_yaw), pitch(r_pitch), 
        fovy(d_fovy), aspect(aspect_ratio), near(near_clip), far(far_clip),
        rotation_speed(rot_speed), movement_speed(mov_speed) {};

    enum Direction {
        FORWARD = 1,
        BACKWARD = 2,
        LEFT = 4,
        RIGHT = 8,
    };

    glm::mat4 get_view_projection(bool translate) const;
    glm::mat4 get_rotation() const;
    void move(float dt, int dir);

    inline glm::vec3 get_position() { return this->position; } 
    inline void set_aspect(float aspect) { this->aspect = aspect; }
    inline void rotate_yaw(float dt) { this->yaw += dt * rotation_speed; }
    inline void rotate_pitch(float dt) { this->pitch += dt * rotation_speed; }
    inline void set_movement_speed(float movement_speed) { this->movement_speed = movement_speed; }
    inline void set_rotation_speed(float rotation_speed) { this->rotation_speed = rotation_speed; };

private: 	
    glm::vec3 position;
    float yaw, pitch;
    float fovy, aspect, near, far;
    
    float rotation_speed;
    float movement_speed; 
};

#endif //WR_CAMERA_H