from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_last_x, g_last_y = 0, 0
g_left_btn = False
g_radius = 10
g_theta = np.radians(45)
g_phi = np.radians(45)
g_target = glm.vec3(0)
g_P = glm.mat4()

g_vertex_shader_src = '''
#version 330 core

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_color;

out vec3 v_color;
uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(vin_pos, 1.0);
    v_color = vin_color;
}
'''

g_fragment_shader_src = '''
#version 330 core
in vec3 v_color;
out vec4 FragColor;

void main() {
    FragColor = vec4(v_color, 1.0);
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
 # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def prepare_vao_cube():
    yellow = [1.0, 0.9, 0.2]
    vertices = glm.array(glm.float32,
        -1, -1,  1, *yellow,  1, -1,  1, *yellow,  1,  1,  1, *yellow,
        -1, -1,  1, *yellow,  1,  1,  1, *yellow, -1,  1,  1, *yellow,
        -1, -1, -1, *yellow, -1,  1, -1, *yellow,  1,  1, -1, *yellow,
        -1, -1, -1, *yellow,  1,  1, -1, *yellow,  1, -1, -1, *yellow,
        -1, -1, -1, *yellow, -1, -1,  1, *yellow, -1,  1,  1, *yellow,
        -1, -1, -1, *yellow, -1,  1,  1, *yellow, -1,  1, -1, *yellow,
         1, -1, -1, *yellow,  1,  1, -1, *yellow,  1,  1,  1, *yellow,
         1, -1, -1, *yellow,  1,  1,  1, *yellow,  1, -1,  1, *yellow,
        -1,  1, -1, *yellow, -1,  1,  1, *yellow,  1,  1,  1, *yellow,
        -1,  1, -1, *yellow,  1,  1,  1, *yellow,  1,  1, -1, *yellow,
        -1, -1, -1, *yellow,  1, -1, -1, *yellow,  1, -1,  1, *yellow,
        -1, -1, -1, *yellow,  1, -1,  1, *yellow, -1, -1,  1, *yellow,
    )
    
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    return VAO, 36 

def prepare_vao_grid():
    grid_lines = []
    color_gray = [0.5, 0.5, 0.5]
    for i in range(-25, 26): #50x50
        grid_lines += [i, 0, -25] + color_gray + [i, 0, 26] + color_gray
        grid_lines += [-25, 0, i] + color_gray + [26, 0, i] + color_gray

    vertices = glm.array(glm.float32, *grid_lines)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)
    return VAO, len(grid_lines) // 6

def prepare_vao_axes():
    axis_lines = [
        -25, 0, 0,   1, 0, 0, #x red
        25, 0, 0,    1, 0, 0,

        0, -25, 0,   0, 0, 1, #y blue
        0, 25, 0,    0, 0, 1,

        0, 0, -25,   0, 1, 0, #z green
        0, 0, 25,    0, 1, 0,
    ]
    vertices = glm.array(glm.float32, *axis_lines)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    return VAO, len(axis_lines) // 6
def cursor_pos_callback(window, xpos, ypos):
    global g_last_x, g_last_y, g_theta, g_phi, g_target, g_radius

    dx = xpos - g_last_x
    dy = ypos - g_last_y
    g_last_x, g_last_y = xpos, ypos

    if not g_left_btn:
        return

    alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS or glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS
    ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS or glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS
    shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS or glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS

    if alt and ctrl: #zoom
        g_radius *= 1.0 + dy * 0.01
        g_radius = np.clip(g_radius, 2, 100)
    elif alt and shift: #pan
        cam_dir = glm.normalize(camera_pos() - g_target)
        right = glm.normalize(glm.cross(cam_dir, glm.vec3(0, 1, 0)))
        up = glm.normalize(glm.cross(right, cam_dir))
        g_target += 0.01 * dx * right + 0.01 * dy * up
    elif alt: #orbit
        g_theta -= np.radians(dx)
        g_phi += np.radians(dy)
        g_phi = np.clip(g_phi, np.radians(10), np.radians(170))
    

def mouse_button_callback(window, button, action, mods):
    global g_left_btn
    if button == GLFW_MOUSE_BUTTON_LEFT:
        g_left_btn = action == GLFW_PRESS

def framebuffer_size_callback(window, width, height):
    global g_P
    glViewport(0, 0, width, height)
    aspect = width / height
    g_P = glm.perspective(glm.radians(45), aspect, 0.1, 100)

def camera_pos():
    sin_phi = np.sin(g_phi)
    pos = glm.vec3(
        g_radius * sin_phi * np.sin(g_theta),
        g_radius * np.cos(g_phi),
        g_radius * sin_phi * np.cos(g_theta)
    )
    return pos + g_target

def main():
    global g_last_x, g_last_y

    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2023049998_proj_1', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
    glfwSetCursorPosCallback(window, cursor_pos_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')

    vao_cube, cube_vertices = prepare_vao_cube()
    vao_grid, grid_vertices = prepare_vao_grid()
    vao_axes, axis_vertices = prepare_vao_axes()
    
    g_last_x, g_last_y = glfwGetCursorPos(window)
    framebuffer_size_callback(window, 800, 800)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glUseProgram(shader_program)
        
        cam_pos = camera_pos()
        V = glm.lookAt(cam_pos, g_target, glm.vec3(0, 1, 0))
        MVP = g_P * V * glm.mat4()

        #cube
        glBindVertexArray(vao_cube)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glDrawArrays(GL_TRIANGLES, 0, cube_vertices)
        #xyz
        glBindVertexArray(vao_axes)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glDrawArrays(GL_LINES, 0, axis_vertices)
        #grid
        glBindVertexArray(vao_grid)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glDrawArrays(GL_LINES, 0, grid_vertices)
        # swap front and back buffers
        glfwSwapBuffers(window)
        # poll events
        glfwPollEvents()
    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()