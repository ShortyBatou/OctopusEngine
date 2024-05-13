#version 330

#ifdef VERTEX_SHADER

layout(std430, binding = 1) buffer layoutName
{
    vec4 colors[];
};

layout(location = 0) in vec3 position;

    uniform mat4 mvp;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec4 f_color;
    void main( )
    {
        f_color = colors[gl_VertexID];
        gl_Position= mvp * vec4(position.x, position.y, position.z, 1);
    }

#endif

#ifdef FRAGMENT_SHADER

    in vec4 f_color;
    out vec4 fragment_color;
    uniform float wireframe_intencity;
    void main( )
    {  
        fragment_color = f_color * wireframe_intencity;
    }
#endif
