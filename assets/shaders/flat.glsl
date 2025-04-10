#version 430

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
        gl_Position= mvp * vec4(position.x, position.y, position.z, 1);
    }

#endif

#ifdef GEOMETRY_SHADER
    layout(triangles) in;
    layout(triangle_strip, max_vertices = 3) out;

    in vec3 g_vert[];

    out vec3 f_vertPosition;
    out vec3 f_normal;

    void main()
    {
        f_normal = normalize( cross( vec3( g_vert[ 1 ] - g_vert[ 0 ] ),
                    vec3( g_vert[ 2 ] - g_vert[ 0 ] ) ) );

        for (int i = 0; i < gl_in.length(); ++i){
            // copy attributes
            gl_Position = gl_in[ i ].gl_Position;
            f_vertPosition = g_vert[ i ];
            EmitVertex();
        }
    }

#endif

#ifdef FRAGMENT_SHADER
    in vec3 f_vertPosition;
    in vec3 f_normal;

    uniform vec4 color = vec4(1, 0, 0, 1);

    const vec3 color_diffuse = vec3( 0.4, 0.4, 0.4 );
    const vec3 lightDir = normalize(vec3(1));
    out vec4 fragment_color;
    void main( )
    {  
        float shade = max(0.3, sqrt(dot(lightDir, f_normal) * .5 + .5));
        fragment_color = vec4(shade * color.rgb, color.a);
    }
#endif