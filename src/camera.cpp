#include "camera.h"

glm::mat4 Camera::get_view_projection(bool translate) const {
    glm::mat4 rotation = get_rotation();
    glm::mat4 translation = glm::mat4(1.0);
    if (translate)
        translation = glm::translate(translation, position);

    glm::mat4 projection = glm::perspective(this->fovy, this->aspect, this->near, this->far);
        return projection * glm::inverse(translation * rotation);
}

glm::mat4 Camera::get_rotation() const {
    glm::mat4 rotate = glm::rotate(glm::mat4(1.0), 
        glm::radians<float>(this->yaw), glm::vec3(0.0, 1.0, 0.0)
    );
    return glm::rotate(rotate, glm::radians<float>(this->pitch), glm::vec3(1.0, 0.0, 0.0));
}

void Camera::move(float dt, int dir) {
    glm::mat4 rotation = get_rotation();
    glm::vec3 forward = glm::vec3(rotation * glm::vec4(0.0, 0.0, -1.0, 0.0));
    glm::vec3 right = glm::vec3(rotation * glm::vec4(1.0, 0.0, 0.0, 0.0));
    
    if (dir & Direction::FORWARD)
        this->position += forward * (float) (dt * this->movement_speed);
    if (dir & Direction::BACKWARD)
        this->position -= forward * (float) (dt * this->movement_speed);
    if (dir & Direction::RIGHT)
        this->position += right * (float) (dt * this->movement_speed);
    if (dir & Direction::LEFT)
        this->position -= right * (float) (dt * this->movement_speed);
}