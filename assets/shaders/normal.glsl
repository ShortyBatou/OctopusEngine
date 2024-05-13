#version 330

#ifdef VERTEX_SHADER

    layout(location = 0) in vec3 position;
    layout(location = 2) in vec3 normal;
    uniform mat4 mvp;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 g_vert;
    out vec3 g_normal;

    void main( )
    {
        g_vert = position;
        g_normal = normal;
    }

#endif

#ifdef GEOMETRY_SHADER
    layout(points) in;
    layout(line_strip, max_vertices = 2) out;

    in vec3 g_vert[];
    in vec3 g_normal[];

    uniform mat4 mvp;
    uniform float n_length = 0.05;

    void main()
    {

        gl_Position = mvp * vec4(g_vert[0], 1);
        EmitVertex();

        gl_Position = mvp * vec4(g_vert[0] + g_normal[0] * n_length, 1);
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