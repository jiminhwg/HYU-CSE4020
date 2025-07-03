from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

g_last_x, g_last_y = 0, 0
g_left_btn = False
g_radius = 10
g_theta = np.radians(45)
g_phi = np.radians(45)
g_target = glm.vec3(0)
g_P = glm.mat4()

bvh_data = None
animate = False
start_time = 0
use_obj_mode = False
joint_obj_models = {}

# ---- from proj 1 & 2
g_vertex_shader_src = '''
#version 330 core

layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_normal;

out vec3 frag_pos;
out vec3 frag_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 world_pos = model * vec4(vin_pos, 1.0);
    frag_pos = vec3(world_pos);
    frag_normal = mat3(transpose(inverse(model))) * vin_normal;

    gl_Position = projection * view * world_pos;
}
'''
g_fragment_shader_src = '''
#version 330 core
in vec3 frag_pos;
in vec3 frag_normal;

out vec4 FragColor;

uniform vec3 light_pos;
uniform vec3 view_pos;

void main() {
    vec3 norm = normalize(frag_normal);
    vec3 light_color = vec3(1, 1, 1);
    vec3 object_color = vec3(1.0, 0.9, 0.2); 
    vec3 light_dir = normalize(light_pos - frag_pos);
    vec3 view_dir = normalize(view_pos - frag_pos);

    vec3 ambient = 0.3 * light_color;
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * light_color;

    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    vec3 specular = 0.3 * spec * light_color;

    vec3 result = (ambient + diffuse + specular) * object_color;
    FragColor = vec4(result, 1.0);
}
'''

grid_vertex_shader_src = '''
#version 330 core
layout(location = 0) in vec3 vin_pos;
layout(location = 1) in vec3 vin_color;

out vec3 frag_color;

uniform mat4 MVP;

void main() {
    frag_color = vin_color;
    gl_Position = MVP * vec4(vin_pos, 1.0);
}
'''
grid_fragment_shader_src = '''
#version 330 core
in vec3 frag_color;
out vec4 FragColor;

void main() {
    FragColor = vec4(frag_color, 1.0);
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
    g_P = glm.perspective(glm.radians(45), aspect, 0.1, 500)

def camera_pos():
    sin_phi = np.sin(g_phi)
    pos = glm.vec3(
        g_radius * sin_phi * np.sin(g_theta),
        g_radius * np.cos(g_phi),
        g_radius * sin_phi * np.cos(g_theta)
    )
    return pos + g_target
# from proj 1 & 2 ----

def key_callback(window, key, scancode, action, mods):
    global animate, use_obj_mode, bvh_data, joint_obj_models
    global current_frame_idx, last_frame_time
    if action == GLFW_PRESS:
        if key == GLFW_KEY_SPACE:
            animate = not animate
            if animate:
                current_frame_idx = 0
                last_frame_time = glfwGetTime()
        elif key == GLFW_KEY_1:
            bvh_path = os.path.join("bvh", "walk.bvh")
            if os.path.exists(bvh_path):
                bvh_data = parse_bvh_file(bvh_path)  
                joint_obj_models = load_joint_objs(bvh_data['flat_joints'])
                use_obj_mode = True

# ---- rendering functions
def create_unit_box_vao():
    cube = [(-.5,-.5,-.5),(.5,-.5,-.5),(.5,.5,-.5),(-.5,.5,-.5), # back
            (-.5,-.5,.5),(.5,-.5,.5),(.5,.5,.5),(-.5,.5,.5)]    # front
    index = [0,1,2,2,3,0, 1,5,6,6,2,1, 5,4,7,7,6,5,
           4,0,3,3,7,4, 3,2,6,6,7,3, 4,5,1,1,0,4]
    verts = []

    for i in index:
        v = cube[i]
        n = glm.normalize(glm.vec3(v))
        verts.extend([*v, *n])

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    arr = np.array(verts, dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, False, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, False, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    return VAO, len(verts)//6

def apply_motion(joint, frame, parent):
    T = glm.mat4(1)
    R = glm.mat4(1)

    channels_lower = [channel.lower() for channel in joint.channels] # consistency

    if 'xposition' in channels_lower:
        index = joint.channel_start_index
        x = frame[index + channels_lower.index('xposition')]
        y = frame[index + channels_lower.index('yposition')]
        z = frame[index + channels_lower.index('zposition')]
        T = glm.translate(glm.mat4(1), glm.vec3(x, y, z))
    else:
        T = glm.translate(glm.mat4(1), joint.offset)

    for i, channel in enumerate(channels_lower):
        val = frame[joint.channel_start_index + i]
        if 'rotation' in channel:
            angle_rad = glm.radians(val)
            if channel == 'xrotation':
                R = R * glm.rotate(glm.mat4(1), angle_rad, glm.vec3(1, 0, 0))
            elif channel == 'yrotation':
                R = R * glm.rotate(glm.mat4(1), angle_rad, glm.vec3(0, 1, 0))
            elif channel == 'zrotation':
                R = R * glm.rotate(glm.mat4(1), angle_rad, glm.vec3(0, 0, 1))

    return parent * T * R

def draw_joint_anim(joint, frame, parent, shader, loc, vao, count):
    current = apply_motion(joint, frame, parent)
   
    for child in joint.children:
        offset = child.offset
        length = glm.length(offset) # length of bone
        if length < 1e-6:
            continue
        
        dir = glm.normalize(offset)
        dot = glm.dot(glm.vec3(0, 1, 0), dir) # aligning bone
        dot = max(min(dot, 1.0), -1.0)
        angle = glm.acos(dot)

        if glm.length(glm.cross(glm.vec3(0, 1, 0), dir)) > 1e-4:
            axis = glm.normalize(glm.cross(glm.vec3(0, 1, 0), dir))
            rot = glm.rotate(glm.mat4(1), angle, axis)
        else:
            rot = glm.mat4(1)

        model_bone = current * rot * glm.translate(glm.mat4(1), glm.vec3(0, 0.5*length, 0)) * glm.scale(glm.mat4(1), glm.vec3(0.3, length, 0.3))
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm.value_ptr(model_bone))
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, count)

        draw_joint_anim(child, frame, current, shader, loc, vao, count)

def draw_joint_obj(joint, frame, parent, shader, loc):
    current = apply_motion(joint, frame, parent)

    left_arm_correction = glm.rotate(glm.mat4(1), glm.radians(70), glm.vec3(0, 0, 1))  # fix position of arms
    right_arm_correction = glm.rotate(glm.mat4(1), glm.radians(-70), glm.vec3(0, 0, 1))

    left_arm_joints = {"LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumb"}
    right_arm_joints = {"RightArm", "RightForeArm", "RightHand", "RightHandThumb"}

    if joint.name in left_arm_joints: #fix
        model_transform = current * left_arm_correction
    elif joint.name in right_arm_joints:
        model_transform = current * right_arm_correction
    else:
        model_transform = current

    if joint.name in joint_obj_models:
        vao, count = joint_obj_models[joint.name]
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm.value_ptr(model_transform))
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, count)

    for child in joint.children:
        draw_joint_obj(child, frame, current, shader, loc)

# rendering functions ----

def drop_callback(window, paths):
    global bvh_data, use_obj_mode
    for path in paths:
        if path.endswith('.bvh'):
            bvh_data = parse_bvh_file(path)
            print_bvh_summary(path, bvh_data)
            use_obj_mode = False

# ---- bvh parser
class Joint:
    def __init__(self, name):
        self.name = name
        self.offset = glm.vec3(0)
        self.channels = []
        self.children = []
        self.parent = None
        self.channel_start_index = 0
        
def parse_bvh_file(path):
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    index = 0

    def parse_joint():
        nonlocal index
        line = lines[index]
        if line.startswith('End'):
            joint = Joint("End Site")
            index += 1  
            index += 1
            scale = 0.1  # scaling factor
            joint.offset = glm.vec3(*[float(v) * scale for v in lines[index].split()[1:4]])       
            index += 1
            index += 1
            return joint

        _, name = line.split()[:2]
        joint = Joint(name)
        index += 1
        index += 1

        # OFFSET
        scale = 0.1  # scaling factor
        joint.offset = glm.vec3(*[float(v) * scale for v in lines[index].split()[1:4]])       
        index += 1

        # CHANNELS
        parts = lines[index].split()
        num_channels = int(parts[1])
        joint.channels = parts[2:2+num_channels]
        index += 1

        # JOINT or End Site
        while index < len(lines) and not lines[index].startswith('}'):
            if lines[index].startswith('JOINT') or lines[index].startswith('End'):
                child = parse_joint()
                child.parent = joint
                joint.children.append(child)
        index += 1
        return joint

    # HIERARCHY 
    index += 1

    root_joint = parse_joint()

    # flatten joints
    flat_joints = []
    def flatten_joint(j):
        if j.name != "End Site":
            flat_joints.append(j)
        for c in j.children:
            flatten_joint(c)
    flatten_joint(root_joint)

    # channel start index
    channel_index = 0
    for j in flat_joints:
        j.channel_start_index = channel_index
        channel_index += len(j.channels)

    # MOTION
    index += 1

    num_frames = int(lines[index].split()[1])
    index += 1

    frame_time = float(lines[index].split()[2])
    index += 1

    frames = []
    for i in range(num_frames):
        frame_vals = list(map(float, lines[index].split()))
        frames.append(frame_vals)
        index += 1
    scale = 0.1  

    root_pos_0 = [v*scale for v in frames[0][:3]] # initial root pos (scaled)

    for frame in frames:
        for i in range(3):
            frame[i] = frame[i] * scale - root_pos_0[i]

    return {
        'root': root_joint,
        'flat_joints': flat_joints,
        'num_frames': num_frames,
        'frame_time': frame_time,
        'frames': frames
    }

def print_bvh_summary(path, data):
    print(f"File name: {os.path.basename(path)}")
    print(f"Number of frames: {data['num_frames']}")
    print(f"FPS: {1.0 / data['frame_time']:.2f}")
    print(f"Number of joints: {len(data['flat_joints'])}")
    print("List of joint names:")
    for j in data['flat_joints']:
        print("  ", j.name)

def load_joint_objs(joint_list):
    models = {}
    base_dir = os.path.abspath(os.path.dirname(__file__))
    obj_dir = os.path.join(base_dir, "obj_files")

    def load_object(path):
        positions = []
        normals = []
        faces = []

        file_name = os.path.basename(path)
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    positions.append(list(map(float, line.split()[1:4])))
                elif line.startswith('vn '):
                    normals.append(list(map(float, line.split()[1:4])))
                elif line.startswith('f '):
                    faces.append(line.split()[1:])

        final_vertices = []
        final_normals = []

        for face in faces:
            newlist = []
            for token in face:
                if '//' in token:  # v//vn format
                    v_str, n_str = token.split('//')
                else:
                    parts = token.split('/')
                    v_str = parts[0]
                    n_str = parts[2] if len(parts) > 2 and parts[2] else '0'
                vi = int(v_str) - 1
                ni = int(n_str) - 1 if n_str.isdigit() else 0
                newlist.append((vi, ni))

            for j in range(1, len(newlist) - 1):
                for k in [0, j, j + 1]:
                    vi, ni = newlist[k]
                    final_vertices.extend(positions[vi])
                    if 0 <= ni < len(normals):
                        final_normals.extend(normals[ni])
                    else:
                        final_normals.extend([0.0, 0.0, 1.0])  # fallback normal

        return final_vertices, final_normals

    for j in joint_list:
        name = j.name
        path = os.path.join(obj_dir, f"{name}.obj")

        verts, norms = load_object(path)

        interleaved = []
        for i in range(0, len(verts), 3):
            v = verts[i:i+3]
            n = norms[i:i+3]
            interleaved.extend(v + n)
        arr = np.array(interleaved, dtype=np.float32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, False, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, False, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        models[name] = (vao, len(verts) // 3)

    return models

# bvh parser ----


def main():
    global g_last_x, g_last_y
    
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    window = glfwCreateWindow(800, 800, 'proj_3', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)
    glfwSetCursorPosCallback(window, cursor_pos_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetDropCallback(window, drop_callback)

    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    model_loc = glGetUniformLocation(shader_program, 'model')
    view_loc = glGetUniformLocation(shader_program, 'view')
    proj_loc = glGetUniformLocation(shader_program, 'projection')
    light_pos_loc = glGetUniformLocation(shader_program, 'light_pos')
    view_pos_loc = glGetUniformLocation(shader_program, 'view_pos')

    box_vao, box_count = create_unit_box_vao()

    grid_shader = load_shaders(grid_vertex_shader_src, grid_fragment_shader_src)
    MVP_loc = glGetUniformLocation(grid_shader, 'MVP')

    vao_grid, grid_vertices = prepare_vao_grid()
    vao_axes, axis_vertices = prepare_vao_axes()
    glfwSetDropCallback(window, drop_callback)

    g_last_x, g_last_y = glfwGetCursorPos(window)
    framebuffer_size_callback(window, 800, 800)

    current_frame_idx = 0
    last_frame_time = glfwGetTime()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # render in "wireframe mode"
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
   
        cam = camera_pos()
        V = glm.lookAt(cam, g_target, glm.vec3(0,1,0))

        #render using grid shader
        glUseProgram(grid_shader)
        MVP = g_P * V * glm.mat4()
        #xyz
        glBindVertexArray(vao_axes)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glDrawArrays(GL_LINES, 0, axis_vertices)
        #grid
        glBindVertexArray(vao_grid)
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glDrawArrays(GL_LINES, 0, grid_vertices)      

        #render using phong shader
        glUseProgram(shader_program)
        glUniform3fv(light_pos_loc, 1, glm.value_ptr(glm.vec3(0,20,30)))
        glUniform3fv(view_pos_loc, 1, glm.value_ptr(cam))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(V))
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(g_P))

        #draw .bvh 
        if bvh_data:
            if animate:
                current_time = glfwGetTime()
                if current_time - last_frame_time >= bvh_data['frame_time']:
                    current_frame_idx = (current_frame_idx+1) % bvh_data['num_frames']
                    last_frame_time = current_time
                frame_idx = current_frame_idx
                
            else:
                frame_idx = 0
            root = bvh_data['root']
            frame = bvh_data['frames'][frame_idx]

            if use_obj_mode:
                draw_joint_obj(root, frame, glm.mat4(1), shader_program, model_loc)
            else:
                draw_joint_anim(root, frame, glm.mat4(1), shader_program, model_loc, box_vao, box_count)
        # swap front and back buffers
        glfwSwapBuffers(window)
        # poll events
        glfwPollEvents()
    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()