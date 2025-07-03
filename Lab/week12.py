from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
from PIL import Image

g_cam_ang = 0.
g_cam_height = .1

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 
layout (location = 2) in vec2 vin_uv; 

out vec3 vout_surface_pos;
out vec3 vout_normal;
out vec2 vout_uv;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
    vout_uv = vin_uv;
}
'''

g_fragment_shader_src = '''
#version 330 core
in vec2 vout_uv;
out vec4 FragColor;

uniform sampler2D texture_diffuse;
uniform vec2 tex_offset;

void main()
{
    vec2 uv = vout_uv - tex_offset;
    FragColor = texture(texture_diffuse, uv);

}
'''

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

def prepare_vao_cube():
    # A rectangle (2 triangles) with proper UVs
    vertices = np.array([
    # positions         # normals       # uvs
     0.0, 0.0, 0.0,      0, 0, 1,        0.0, 0.0,
     2.0, 0.0, 0.0,      0, 0, 1,        1.0, 0.0,
     2.0, 1.0, 0.0,      0, 0, 1,        1.0, 1.0,

     0.0, 0.0, 0.0,      0, 0, 1,        0.0, 0.0,
     2.0, 1.0, 0.0,      0, 0, 1,        1.0, 1.0,
     0.0, 1.0, 0.0,      0, 0, 1,        0.0, 1.0,
], dtype=np.float32)




    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)
    # uv
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    return vao


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
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    loc_MVP = glGetUniformLocation(shader_program, 'MVP')
    loc_M = glGetUniformLocation(shader_program, 'M')
    loc_view_pos = glGetUniformLocation(shader_program, 'view_pos')
    loc_tex_offset = glGetUniformLocation(shader_program, 'tex_offset')

    # prepare vaos
    vao_cube = prepare_vao_cube()


    glUseProgram(shader_program)

    ############################################
    # diffuse texture

    # create texture
    texture_diffuse = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_diffuse)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)
    # set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    try:
        img = Image.open('./320px-Solarsystemscope_texture_8k_earth_daymap.jpg')
        
        # vertically filp the image 
        # because OpenGL expects 0.0 on y-axis to be on the bottom edge, but images usually have 0.0 at the top of the y-axis
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # glTexImage2D(target, level, texture internalformat, width, height, border, image data format, image data type, data)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    
        # generate mipmaps
        glGenerateMipmap(GL_TEXTURE_2D)

        img.close()

    except:
        print("Failed to load texture")

    ############################################
    # specular texture

    # create texture
    texture_specular = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_specular)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT)

    # set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    try:
        img = Image.open('./plain-checkerboard.jpg')
        # img = Image.open('./320px-Solarsystemscope_texture_8k_earth_daymap-grayscale.jpg')
        
        # vertically filp the image 
        # because OpenGL expects 0.0 on y-axis to be on the bottom edge, but images usually have 0.0 at the top of the y-axis
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # glTexImage2D(target, level, texture internalformat, width, height, border, image data format, image data type, data)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
    
        # generate mipmaps
        glGenerateMipmap(GL_TEXTURE_2D)

        img.close()

    except:
        print("Failed to load texture")

    ############################################

    # for i-th texture unit, sampler uniform variable value should be i
    glUniform1i(glGetUniformLocation(shader_program, 'texture_diffuse'), 0)
    # activate i-th texture unit by passing GL_TEXTUREi
    glActiveTexture(GL_TEXTURE0)  
    # texture object is binded on this activated texture unit
    glBindTexture(GL_TEXTURE_2D, texture_diffuse)


    #glUniform1i(glGetUniformLocation(shader_program, 'texture_specular'), 1)
    #glActiveTexture(GL_TEXTURE1)
    #glBindTexture(GL_TEXTURE_2D, texture_specular)


    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # projection matrix
        P = glm.perspective(45, 1, 1, 20)

        # view matrix
        view_pos = glm.vec3(2.5*np.sin(g_cam_ang),g_cam_height,2.5*np.cos(g_cam_ang))
        V = glm.lookAt(view_pos, glm.vec3(0,0,0), glm.vec3(0,1,0))


        # animating
        t = glfwGetTime()

        # rotation
        th = np.radians(t*90)
        R = glm.rotate(th, glm.vec3(0,1,0))

        M = glm.mat4(1.0)
        # # try applying rotation
        # M = R

        # update uniforms
        MVP = P*V*M
        glUseProgram(shader_program)
        glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniformMatrix4fv(loc_M, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(loc_view_pos, view_pos.x, view_pos.y, view_pos.z)

        offset_x = (glfwGetTime()*0.7) 
        glUniform2f(loc_tex_offset, offset_x, 0.0)

        
        # draw cube w.r.t. the current frame MVP
        glBindVertexArray(vao_cube)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()