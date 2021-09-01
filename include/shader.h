#ifndef WR_SHADER_H
#define WR_SHADER_H

/**
 * 
 * Temporary file for storing shaders not to bloat main
 * 
 */

const char* texture_vs_code = R"(
#version 410 core
layout(location = 0) in vec2 a_Pos;
layout(location = 1) in vec2 a_TextureData;

layout(location = 0) out vec2 vs_UV;

void main() {
  gl_Position = vec4(a_Pos, 0.0, 1.0); 
}
)";

const char* texture_fs_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec2 vs_UV;

uniform sampler2D texture0;

void main() {
  color = texture(texture0, vs_UV);
}
)";

const char* skybox_vs_code = R"(
#version 410 core
layout(location = 0) in vec3 a_Pos;

layout(location = 0) out vec3 vs_TexCoord;

uniform mat4 u_ViewProjection;

void main() {
  vs_TexCoord = a_Pos;
  vec4 proj_pos = u_ViewProjection * vec4(a_Pos, 1.0);
  gl_Position = proj_pos.xyww;  
}
)";

const char* skybox_fs_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec3 vs_TexCoord;

uniform samplerCube cube_map;

void main() {
  color = texture(cube_map, vs_TexCoord);
}
)";

const char* water_vs_code = R"(
#version 410 core
layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec3 a_Color;
layout(location = 2) in vec3 a_Normal;
layout(location = 3) in vec2 a_UV;

layout(location = 0) out vec3 vs_Color;
layout(location = 1) out vec3 vs_Normal;
layout(location = 2) out vec3 vs_LightSourceDir;
layout(location = 3) out vec3 vs_CameraDir;

uniform mat4 u_ViewProjection;
uniform mat4 u_Model;
uniform vec3 u_CameraPos;

void main()
{
  // Constants
  vec3 lightPos = vec3(20.0, 30.0, -30.0);
  
  vec4 m_Pos = u_Model * vec4(a_Pos, 1.0);
 
  vs_Color = a_Color;
  vs_Normal = normalize(transpose(inverse(mat3(u_Model))) * a_Normal);
  vs_LightSourceDir = normalize(lightPos - m_Pos.xyz);
  vs_CameraDir = normalize(u_CameraPos - m_Pos.xyz);

  gl_Position = u_ViewProjection * m_Pos; 
}
)";

const char* cubemap_water_fs_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec3 vs_Color;
layout(location = 1) in vec3 vs_Normal;
layout(location = 2) in vec3 vs_LightSourceDir;
layout(location = 3) in vec3 vs_CameraDir;

uniform samplerCube cube_map;

// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
// Tessendorf 4.3 - Building a Shader for renderman
// @param I Incident vector
// @param N Normal vector
// @param ior Index of refraction
float reflectivity(vec3 I, vec3 N, float ior) {
  float costhetai = abs(dot(normalize(I), normalize(N)));
  float thetai = acos(costhetai);
  float sinthetat = sin(thetai) / ior;
  float thetat = asin(sinthetat);
  if(thetai == 0.0) {
    float reflectivity = (ior - 1)/(ior + 1);
    return reflectivity * reflectivity;
  } else {
    float fs = sin(thetat - thetai) / sin(thetat + thetai);
    float ts = tan(thetat - thetai) / tan(thetat + thetai);
    return 0.5 * ( fs*fs + ts*ts );
  } 
}

void main() { 
  float refraction_index = 1.0 / 1.33;
  vec3 refractionDir = refract(-vs_CameraDir, vs_Normal, refraction_index);
  vec3 reflectionDir = reflect(-vs_CameraDir, vs_Normal);
  float reflectivity = reflectivity(vs_CameraDir, vs_Normal, 1.0 / refraction_index);

  // Intensities
  vec3 i_reflect = texture(cube_map, reflectionDir).xyz; 
  vec3 i_refract = vs_Color;
  if (refractionDir != vec3(0.0)) // If refractionDir is 0-vector, something is wrong. 
    i_refract = texture(cube_map, refractionDir).xyz;

  // Blinn-Phong illumination using half-way vector instead of reflection.
  vec3 halfwayDir = normalize(vs_LightSourceDir + vs_CameraDir);
  float specular = pow(max(dot(vs_Normal, halfwayDir), 0.0), 20.0);
  
  const vec3 light_color = 0.4 * normalize(vec3(253, 251, 211));
  
  vec3 reflection_refraction = reflectivity * i_reflect + (1 - reflectivity) * i_refract;
  color = vec4(reflection_refraction + light_color * specular, 1.0);
}
)";

const char* color_water_fs_code = R"(
#version 410 core
out vec4 color;

layout(location = 0) in vec3 vs_Color;
layout(location = 1) in vec3 vs_Normal;
layout(location = 2) in vec3 vs_LightSourceDir;
layout(location = 3) in vec3 vs_CameraDir;

void main() { 
  // Blinn-Phong illumination using half-way vector instead of reflection.
  vec3 halfwayDir = normalize(vs_LightSourceDir + vs_CameraDir);
  float specular = pow(max(dot(vs_Normal, halfwayDir), 0.0), 10.0);
  float diffuse = dot(vs_LightSourceDir, vs_Normal);
  
  const vec3 light_color = 0.4 * normalize(vec3(253, 251, 211));
  
  color = vec4(vs_Color * diffuse + light_color * specular, 1.0);
}
)";

#endif // WR_SHADERS_H