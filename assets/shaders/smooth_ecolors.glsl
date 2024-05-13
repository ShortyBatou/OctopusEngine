#version 330

#ifdef VERTEX_SHADER

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normal;

    uniform mat4 mvp;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec4 f_color;
    out vec3 f_normal;
    void main( )
    {
        f_color = color;
        f_normal = normal;
        gl_Position= mvp * vec4(position.x, position.y, position.z, 1);
    }

#endif

#ifdef FRAGMENT_SHADER

    in vec3 f_normal;
    in vec4 f_color;

    const vec3 color_diffuse = vec3( 0.4, 0.4, 0.4 );
    const vec3 lightDir = normalize(vec3(1));

    out vec4 fragment_color;
    void main( )
    {  
        float shade = max(0.3, sqrt(dot(lightDir, f_normal) * .5 + .5));
        fragment_color = vec4(shade * f_color.rgb, f_color.a);
    }
#endif
