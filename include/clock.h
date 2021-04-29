#ifndef WR_CLOCK_H
#define WR_CLOCK_H

#include <GLFW/glfw3.h>

/**
 * Clock structure based on glfw 
 * 
 * Requires an initialized glfw context
 */
struct Clock {
public:
    Clock() : current_tick(glfwGetTime ()), last_tick(current_tick), start_tick(current_tick) {}

    // Returns the time in seconds since last call to `tick`.
    float tick () {
        this->last_tick    = this->current_tick;
        this->current_tick = glfwGetTime();

        return this->current_tick - this->last_tick;
    }

    float since_start () {
        return glfwGetTime() - start_tick;
    }

private:
    double current_tick;
    double last_tick;
    const double start_tick;
};

#endif //WR_CLOCK_H