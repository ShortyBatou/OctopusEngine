#version 330

#ifdef VERTEX_SHADER

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

    uniform mat4 mvp;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    out vec4 g_color;
    out vec3 g_vert;
    void main( )
    {
        g_color = color;
        g_vert = position;
        gl_Position= mvp * vec4(position.x, position.y, position.z, 1);
    }

#endif

#ifdef GEOMETRY_SHADER
    layout(triangles) in;
    layout(triangle_strip, max_vertices = 3) out;

    in vec4 g_color[];
    in vec3 g_vert[];

    out vec3 f_vertPosition;
    out vec3 f_normal;
    out vec4 f_color;

    void main()
    {
        f_normal = normalize( cross( vec3( g_vert[ 1 ] - g_vert[ 0 ] ),
                    vec3( g_vert[ 2 ] - g_vert[ 0 ] ) ) );

        for (int i = 0; i < gl_in.length(); ++i){
            // copy attributes
            gl_Position = gl_in[ i ].gl_Position;
            f_vertPosition = g_vert[ i ];
            f_color = g_color[i];
            EmitVertex();
        }
    }

#endif

#ifdef FRAGMENT_SHADER
    in vec3 f_vertPosition;
    in vec3 f_normal;
    in vec4 f_color;
    const vec3 lightPos = vec3( 0.0, 0.0, 0.0 );
    const vec4 ia = vec4( 0.6, 0.6, 0.6, 1.0 );
    const vec4 id = vec4( 0.4, 0.4, 0.4, 1.0 );
    const vec4 color_diffuse = vec4( 0.6, 0.6, 0.6, 1.0 );

    out vec4 fragment_color;
    void main( )
    {  
        vec3 normalInterp = normalize( f_normal );
        vec3 lightDir = normalize( lightPos - f_vertPosition );
        float lambertian = abs( dot( lightDir, f_normal ) );
        fragment_color = lambertian * id * color_diffuse + ia * f_color;
    }
#endif
