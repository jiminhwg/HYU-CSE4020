from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_cam_ang = 0.
g_cam_height = .1

g_vertex_shader_src_color_attribute = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_vertex_shader_src_color_uniform = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 

out vec4 vout_color;

uniform mat4 MVP;
uniform vec3 color;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(color, 1.);
    //vout_color = vec4(1,1,1,1);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

class Node:
    def __init__(self, parent, link_transform_from_parent, shape_transform, color):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.link_transform_from_parent = link_transform_from_parent
        self.joint_transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

    def set_joint_transform(self, joint_transform):
        self.joint_transform = joint_transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform_from_parent * self.joint_transform
        else:
            self.global_transform = self.link_transform_from_parent * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color


def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
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


def key_callback(window, key, scancode, action, mods):
    global g_cam_ang, g_cam_height
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    else:
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_1:
                g_cam_ang += np.radians(-10)
            elif key==GLFW_KEY_3:
                g_cam_ang += np.radians(10)
            elif key==GLFW_KEY_2:
                g_cam_height += .1
            elif key==GLFW_KEY_W:
                g_cam_height += -.1

def prepare_vao_box():
    # prepare vertex data (in main memory)
    # 6 vertices for 2 triangles
    vertices = glm.array(glm.float32,
        # position         
        -1 ,  1 ,  0 , # v0
         1 , -1 ,  0 , # v2
         1 ,  1 ,  0 , # v1

        -1 ,  1 ,  0 , # v0
        -1 , -1 ,  0 , # v3
         1 , -1 ,  0 , # v2
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position # color
         0, 0, 0,  1, 0, 0, # x-axis start
         1, 0, 0,  1, 0, 0, # x-axis end 
         0, 0, 0,  0, 1, 0, # y-axis start
         0, 1, 0,  0, 1, 0, # y-axis end 
         0, 0, 0,  0, 0, 1, # z-axis start
         0, 0, 1,  0, 0, 1, # z-axis end 
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def draw_frame(vao, MVP, MVP_loc):
    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 6)

def draw_node(vao, node, VP, MVP_loc, color_loc):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawArrays(GL_TRIANGLES, 0, 6)


def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, '2023049998', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback);

    # load shaders
    shader_for_frame = load_shaders(g_vertex_shader_src_color_attribute, g_fragment_shader_src)
    shader_for_box = load_shaders(g_vertex_shader_src_color_uniform, g_fragment_shader_src)

    # get uniform locations
    loc_MVP_frame = glGetUniformLocation(shader_for_frame, 'MVP')
    loc_MVP_box = glGetUniformLocation(shader_for_box, 'MVP')
    loc_color_box = glGetUniformLocation(shader_for_box, 'color')
    
    # prepare vaos
    vao_box = prepare_vao_box()
    vao_frame = prepare_vao_frame()

    # create a hirarchical model - Node(parent, link_transform_from_parent, shape_transform, color)
    base = Node(None, glm.mat4(), glm.scale((.2,.2,0.)), glm.vec3(0,0,1))
    #arm = Node(base, glm.translate(glm.vec3(.2,0,0)), glm.translate((.5,0,.01)) * glm.scale((.5,.1,0.)), glm.vec3(1,0,0))
    
    red_armL = Node(base, glm.translate(glm.vec3(-.2,0,0)), glm.translate((-.25, 0, .01)) * glm.scale((.25, .1, 0.)), glm.vec3(1,0,0))
    green_armL = Node(red_armL,  glm.translate(glm.vec3(-1,0,0)), glm.translate((-0.25, 0, .02)) * glm.scale((-.25, .1, 0.)) , glm.vec3(0,1,0))
    
    red_armR = Node(base, glm.translate(glm.vec3(.2,0,0)), glm.translate((.25, 0, .01)) * glm.scale((.25, .1, 0.)), glm.vec3(1,0,0))
    green_armR = Node(red_armR, glm.translate(glm.vec3(0,0,0)), glm.translate((0.25, 0, .02)) * glm.scale((.25, .1, 0.)), glm.vec3(0,1,0))

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # projection matrix
        P = glm.ortho(-1,1,-1,1,-10,10)

        # view matrix
        V = glm.lookAt(glm.vec3(1*np.sin(g_cam_ang),g_cam_height,1*np.cos(g_cam_ang)), glm.vec3(0,0,0), glm.vec3(0,1,0))

        # draw world frame
        glUseProgram(shader_for_frame)
        draw_frame(vao_frame, P*V*glm.mat4(), loc_MVP_frame)


        t = glfwGetTime()

        # set local transformations of each node
        base.set_joint_transform(glm.translate((glm.sin(t),0,0)))
        
        #arm.set_joint_transform(glm.rotate(t, (0,0,1)))
        red_armL.set_joint_transform(glm.rotate(-t, (0,0,1)))
        green_armL.set_joint_transform(glm.translate((0.5, 0, 0)) * glm.rotate(-t, (0,0,1)))
        red_armR.set_joint_transform(glm.rotate(t, (0,0,1)))
        green_armR.set_joint_transform(glm.translate((0.5, 0, 0)) * glm.rotate(t, (0,0,1)))

        # recursively update global transformations of all nodes
        base.update_tree_global_transform()

        # draw nodes
        glUseProgram(shader_for_box)
        draw_node(vao_box, base, P*V, loc_MVP_box, loc_color_box)
        draw_node(vao_box, red_armL, P*V, loc_MVP_box, loc_color_box)
        draw_node(vao_box, green_armL, P*V, loc_MVP_box, loc_color_box)
        draw_node(vao_box, red_armR, P*V, loc_MVP_box, loc_color_box)
        draw_node(vao_box, green_armR, P*V, loc_MVP_box, loc_color_box)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
