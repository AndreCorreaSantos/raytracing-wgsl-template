const THREAD_COUNT = 16;
const RAY_TMIN = 0.0001;
const RAY_TMAX = 100.0;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_2_PI = 1.5707964f;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(0) @binding(1)
  var<storage, read_write> rtfb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> spheresb : array<sphere>;

@group(2) @binding(1)
  var<storage, read_write> quadsb : array<quad>;

@group(2) @binding(2)
  var<storage, read_write> boxesb : array<box>;

@group(2) @binding(3)
  var<storage, read_write> trianglesb : array<triangle>;

@group(2) @binding(4)
  var<storage, read_write> meshb : array<mesh>;

struct ray {
  origin : vec3f,
  direction : vec3f,
};

struct sphere {
  transform : vec4f,
  color : vec4f,
  material : vec4f,
};

struct quad {
  Q : vec4f,
  u : vec4f,
  v : vec4f,
  color : vec4f,
  material : vec4f,
};

struct box {
  center : vec4f,
  radius : vec4f,
  rotation: vec4f,
  color : vec4f,
  material : vec4f,
};

struct triangle {
  v0 : vec4f,
  v1 : vec4f,
  v2 : vec4f,
};

struct mesh {
  transform : vec4f, // how do i apply transform, scale and rotation to the obj?
  scale : vec4f,
  rotation : vec4f,
  color : vec4f,
  material : vec4f,
  min : vec4f,  // min and max?
  max : vec4f,
  show_bb : f32,
  start : f32, // index of first and last triangle in triangles buffer? COME BACK LATER
  end : f32, 
};

struct material_behaviour {
  scatter : bool,
  direction : vec3f,
};

struct camera {
  origin : vec3f,
  lower_left_corner : vec3f,
  horizontal : vec3f,
  vertical : vec3f,
  u : vec3f,
  v : vec3f,
  w : vec3f,
  lens_radius : f32,
};

struct hit_record {
  t : f32,
  p : vec3f,
  normal : vec3f,
  object_color : vec4f,
  object_material : vec4f,
  frontface : bool,
  hit_anything : bool,
};

fn ray_at(r: ray, t: f32) -> vec3f
{
  return r.origin + t * r.direction;
}

fn get_ray(cam: camera, uv: vec2f, rng_state: ptr<function, u32>) -> ray
{
  var rd = cam.lens_radius * rng_next_vec3_in_unit_disk(rng_state);
  var offset = cam.u * rd.x + cam.v * rd.y;
  return ray(cam.origin + offset, normalize(cam.lower_left_corner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin - offset));
}

fn get_camera(lookfrom: vec3f, lookat: vec3f, vup: vec3f, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> camera
{
  var camera = camera();
  camera.lens_radius = aperture / 2.0;

  var theta = degrees_to_radians(vfov);
  var h = tan(theta / 2.0);
  var w = aspect_ratio * h;

  camera.origin = lookfrom;
  camera.w = normalize(lookfrom - lookat);
  camera.u = normalize(cross(vup, camera.w));
  camera.v = cross(camera.u, camera.w);

  camera.lower_left_corner = camera.origin - w * focus_dist * camera.u - h * focus_dist * camera.v - focus_dist * camera.w;
  camera.horizontal = 2.0 * w * focus_dist * camera.u;
  camera.vertical = 2.0 * h * focus_dist * camera.v;

  return camera;
}

fn environment_color(direction: vec3f, color1: vec3f, color2: vec3f) -> vec3f
{
  var unit_direction = normalize(direction);
  var t = 0.5 * (unit_direction.y + 1.0);
  var col = (1.0 - t) * color1 + t * color2;

  var sun_direction = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  var sun_color = int_to_rgb(i32(uniforms[17]));
  var sun_intensity = uniforms[16];
  var sun_size = uniforms[18];

  var sun = clamp(dot(sun_direction, unit_direction), 0.0, 1.0);
  col += sun_color * max(0, (pow(sun, sun_size) * sun_intensity));

  return col;
}

fn check_ray_collision(r: ray, max: f32) -> hit_record
{
  var spheresCount = i32(uniforms[19]);
  var quadsCount = i32(uniforms[20]);
  var boxesCount = i32(uniforms[21]);
  var trianglesCount = i32(uniforms[22]);
  var meshCount = i32(uniforms[27]);
  
  var r_ = r;
  var closest = hit_record(RAY_TMAX, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
  var min_t = RAY_TMAX;
  // spheres
  for (var i = 0; i < spheresCount; i++)
  {
    var sp_ = spheresb[i];
    var rec_ = hit_record(RAY_TMAX, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
    
    hit_sphere(sp_.transform.xyz,sp_.transform.w,r_,&rec_,min_t);
    if(rec_.hit_anything)
    {
      min_t = rec_.t;
      rec_.object_color = sp_.color; // object's color is assigned here
      rec_.object_material = sp_.material; // smoothness, absorption, specular, refraction are assigned here.
      closest = rec_;
    }
  }
  // quads
  for (var i = 0; i<quadsCount; i++ )
  {
    var quad_ =  quadsb[i];
    var rec_ = hit_record(RAY_TMAX, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
    hit_quad(r_,quad_.Q,quad_.u,quad_.v,&rec_,min_t);
    if(rec_.hit_anything)
    {
      min_t = rec_.t;
      rec_.object_color = quad_.color;
      rec_.object_material = quad_.material;
      closest = rec_;
    }
  }
  // boxes
  for (var i = 0; i<boxesCount; i++ )
  {
    var box_ =  boxesb[i];
    var rec_ = hit_record(RAY_TMAX, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
    hit_box(r_,box_.center.xyz,box_.radius.xyz,&rec_,min_t);
    if(rec_.hit_anything)
    {
      min_t = rec_.t;
      rec_.object_color = box_.color;
      rec_.object_material = box_.material;
      closest = rec_;
    }
  }
  // meshes
  for (var i =0; i<meshCount; i++ )
  {
    var mesh_ = meshb[i];
    // if(mesh.show_bb == 0.0) {
    //       continue;
    // }

    let ti = i32(mesh_.start);
    let tf = i32(mesh_.end);
    
    for (var j = ti; j<= tf; j++)
    {
      var tri_ =  trianglesb[j];
      var rec_ = hit_record(RAY_TMAX, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
      hit_triangle(r_,tri_.v0.xyz,tri_.v1.xyz,tri_.v2.xyz,&rec_,min_t);
      if(rec_.hit_anything)
      {
        min_t = rec_.t;
        rec_.object_color = mesh_.color;
        rec_.object_material = mesh_.material;
        closest = rec_;
      }
    }
  }



  return closest;
}

fn lambertian(normal: vec3f, absorption: f32, random_sphere: vec3f, rng_state: ptr<function, u32>) -> material_behaviour
{
    var scatter_direction = normal + random_sphere;
    
    if (length(scatter_direction) < 0.001) {
        scatter_direction = normal;
    }

    return material_behaviour(true, normalize(scatter_direction));
}

fn metal(normal: vec3f, direction: vec3f, fuzz: f32, random_sphere: vec3f) -> material_behaviour {
    var r = reflect(normalize(direction), normal);
    var scat = r; //+ fuzz * random_sphere;
    
    if (length(scat) < 0.001) {
        scat = r;
    }
    
    var should_scatter = dot(scat, normal) > 0.0;
    return material_behaviour(true, normalize(scat));
}

fn dielectric(normal : vec3f, r_direction: vec3f, refraction_index: f32, frontface: bool, random_sphere: vec3f, fuzz: f32, rng_state: ptr<function, u32>) -> material_behaviour
{  
  return metal(normal, r_direction, fuzz, random_sphere);
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
    var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

fn emissive(color: vec3f, light: f32) -> material_behaviour
{
  return material_behaviour(true, color*light);
}

fn trace(r: ray, rng_state: ptr<function, u32>) -> vec3f
{
  var maxbounces = i32(uniforms[2]);
  var light = vec3f(0.0);
  var color = vec3f(1.0);
  var r_ = r;
  
  var backgroundcolor1 = int_to_rgb(i32(uniforms[11]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[12]));
  var behaviour = material_behaviour(true, vec3f(0.0));

  //color *= environment_color(r_.direction, backgroundcolor1, backgroundcolor2);
  //var t = 3;
  for (var j = 0; j < maxbounces; j = j + 1)
  {
    // create new ray with each bounce
    var record = check_ray_collision(r_, RAY_TMAX);

    if (!record.hit_anything)
    {
      color *= environment_color(r_.direction, backgroundcolor1, backgroundcolor2);
      light += color;
      break;
    }
    var smoothness = record.object_material.x;
    var absorption = record.object_material.y;
    var specular = record.object_material.z;
    var emission = record.object_material.w;
    var fuzz = 0.0; // FOR NOW DEFINING FUZZ AS 0
    if (emission > 0.0)
    {
      // emissive material is a light source
      var emissive_color = emissive(record.object_color.xyz, emission);
      light += color*emissive_color.direction; // returning light color in the direction of material behaviour
      break; // ray has reached a light source, no more bounces?
    } 

    var rng = rng_next_float(rng_state);
    var rng_sphere = rng_next_vec3_in_unit_sphere(rng_state);

    if (smoothness > 0.0)
    {
      if(specular > rng)
      { // if reflection, we don't change color, we just set the behaviour
        behaviour = metal(record.normal, r_.direction, fuzz,rng_sphere );
      }
    }
    // else if (smoothness < 0.0) // dielectric material, don't know how it works yet shits crazy
    // {
    //   behaviour = dielectric(record.normal, r_.direction, smoothness, record.frontface, rng_sphere, absorption, rng_state);
    // }
    else
    {
        // if smoothness is 0 it's a perfect lambertian, diffuse, material
      behaviour = lambertian(record.normal, absorption, rng_sphere, rng_state);
      color *= record.object_color.xyz * (1.0-absorption);
    }
  
    r_ = ray(record.p + record.normal*0.001, behaviour.direction);
    if (behaviour.scatter == false)
    {
      break;
    }
    
  }
  
  return light;
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
    var rez = uniforms[1];
    var time = u32(uniforms[0]);

    // init_rng (random number generator) we pass the pixel position, resolution and frame
    var rng_state = init_rng(vec2(id.x, id.y), vec2(u32(rez)), time);

    // Get uv
    var fragCoord = vec2f(f32(id.x), f32(id.y));
    var uv = (fragCoord + sample_square(&rng_state)) / vec2(rez);

    // Camera
    var lookfrom = vec3(uniforms[7], uniforms[8], uniforms[9]);
    var lookat = vec3(uniforms[23], uniforms[24], uniforms[25]);

    // Get camera
    var cam = get_camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0), uniforms[10], 1.0, uniforms[6], uniforms[5]);
    var samples_per_pixel = i32(uniforms[4]);

    var color = vec3(0.0, 0.0, 0.0);

    // Steps:
    // 1. Loop for each sample per pixel
    
    for (var i = 0; i<samples_per_pixel; i = i+1)
    {
      var u = (f32(id.x) + rng_next_float(&rng_state)) / f32(rez); // u + noise
      var v = (f32(id.y) + rng_next_float(&rng_state)) / f32(rez); // v + noise
      // 2. Get ray
      var r = get_ray(cam, vec2f(u, v), &rng_state);
      // 3. Call trace function
      color += trace(r, &rng_state);
    }
    color /= f32(samples_per_pixel);
    
    
    // 4. Average the color

    var color_out = vec4(linear_to_gamma(color), 1.0);
    var map_fb = mapfb(id.xy, rez);

    color_out = clamp(color_out, vec4(0.0), vec4(1.0));
    // 5. Accumulate the color
    var should_accumulate = uniforms[3];
    var accumulated_color = rtfb[map_fb]*should_accumulate + color_out;
    
    // Set the color to the framebuffer
    rtfb[map_fb] = accumulated_color;
    fb[map_fb] = accumulated_color/accumulated_color.w;
}