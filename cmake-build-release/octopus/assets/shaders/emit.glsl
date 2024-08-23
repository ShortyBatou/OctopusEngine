#version 430

#ifdef VERTEX_SHADER

    layout(location = 0) in vec3 position;
    uniform mat4 mvp;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main( )
    {
        gl_Position= mvp * vec4(position.x, position.y, position.z, 1);
    }

#endif


#ifdef FRAGMENT_SHADER


    uniform vec4 color = vec4(1, 0, 0, 1);
    
    out vec4 fragment_color;
    void main( )
    {  
        fragment_color = color;
    }
#endif