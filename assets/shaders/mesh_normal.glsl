#version 330

#ifdef VERTEX_SHADER

    layout(location = 0) in vec3 position;
    uniform mat4 mvp;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 g_vert;

    void main( )
    {
        g_vert = position;
    }

#endif

#ifdef GEOMETRY_SHADER
    layout(triangles) in;
    layout(line_strip, max_vertices = 2) out;

    in vec3 g_vert[];

    uniform mat4 mvp;
    uniform float n_length = 0.05;

    void main()
    {
        vec3 _normal = normalize( cross( vec3( g_vert[ 1 ] - g_vert[ 0 ] ),
                    vec3( g_vert[ 2 ] - g_vert[ 0 ] ) ) );

        vec3 c = (g_vert[ 0 ] + g_vert[ 1 ] + g_vert[ 2 ]) * 0.3333333333333; 
        gl_Position = mvp * vec4(c.x, c.y, c.z, 1);
        EmitVertex();

        c = c + _normal * n_length;
        gl_Position = mvp * vec4(c.x, c.y, c.z, 1);
        EmitVertex();
    }

#endif

#ifdef FRAGMENT_SHADER
    uniform vec4 color = vec4(0, 0, 1, 1);

    out vec4 fragment_color;
    void main( )
    {  
        fragment_color = color;
    }
#endif