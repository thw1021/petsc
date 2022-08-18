// Loads finite element descriptions from PETSc's SAWs interface and plot aspects using plotly

var fe_data = {};                                     // data for all loaded finite elements
var state = {};                                       // state of the plot: 
state.selected_element = {};                          // - which element is plotted
state.trace = {};                                     // - the plotly traces
state.trace_offsets = {refel: 0, dual: 0, shape: 0};  // - the offsets for different types of data

const petsc_logo_blue = "#305e89";
// Pink-Yellow Green diverging colorspace
const PiYG = [
  ["0.000000000", "rgb(142,   1,  82)"],
  ["0.003921569", "rgb(144,   2,  83)"],
  ["0.007843138", "rgb(146,   3,  85)"],
  ["0.011764706", "rgb(148,   4,  87)"],
  ["0.015686275", "rgb(150,   5,  88)"],
  ["0.019607844", "rgb(152,   6,  90)"],
  ["0.023529412", "rgb(154,   7,  92)"],
  ["0.027450980", "rgb(157,   8,  93)"],
  ["0.031372550", "rgb(159,   9,  95)"],
  ["0.035294120", "rgb(161,  10,  97)"],
  ["0.039215688", "rgb(163,  11,  98)"],
  ["0.043137256", "rgb(165,  12, 100)"],
  ["0.047058824", "rgb(167,  13, 102)"],
  ["0.050980393", "rgb(170,  14, 103)"],
  ["0.054901960", "rgb(172,  15, 105)"],
  ["0.058823530", "rgb(174,  16, 107)"],
  ["0.062745100", "rgb(176,  17, 108)"],
  ["0.066666670", "rgb(178,  18, 110)"],
  ["0.070588240", "rgb(180,  19, 112)"],
  ["0.074509810", "rgb(182,  20, 114)"],
  ["0.078431375", "rgb(185,  21, 115)"],
  ["0.082352940", "rgb(187,  22, 117)"],
  ["0.086274510", "rgb(189,  23, 119)"],
  ["0.090196080", "rgb(191,  24, 120)"],
  ["0.094117650", "rgb(193,  25, 122)"],
  ["0.098039220", "rgb(195,  26, 124)"],
  ["0.101960786", "rgb(197,  28, 125)"],
  ["0.105882354", "rgb(198,  32, 127)"],
  ["0.109803920", "rgb(199,  36, 129)"],
  ["0.113725490", "rgb(200,  39, 131)"],
  ["0.117647060", "rgb(201,  43, 133)"],
  ["0.121568630", "rgb(202,  46, 135)"],
  ["0.125490200", "rgb(203,  50, 137)"],
  ["0.129411770", "rgb(204,  54, 139)"],
  ["0.133333340", "rgb(205,  57, 141)"],
  ["0.137254910", "rgb(206,  61, 143)"],
  ["0.141176480", "rgb(207,  64, 145)"],
  ["0.145098050", "rgb(208,  68, 147)"],
  ["0.149019610", "rgb(209,  72, 149)"],
  ["0.152941180", "rgb(210,  75, 150)"],
  ["0.156862750", "rgb(211,  79, 152)"],
  ["0.160784320", "rgb(212,  82, 154)"],
  ["0.164705890", "rgb(213,  86, 156)"],
  ["0.168627460", "rgb(214,  90, 158)"],
  ["0.172549020", "rgb(215,  93, 160)"],
  ["0.176470600", "rgb(216,  97, 162)"],
  ["0.180392160", "rgb(217, 100, 164)"],
  ["0.184313730", "rgb(218, 104, 166)"],
  ["0.188235300", "rgb(219, 108, 168)"],
  ["0.192156870", "rgb(220, 111, 170)"],
  ["0.196078430", "rgb(221, 115, 172)"],
  ["0.200000000", "rgb(222, 119, 174)"],
  ["0.203921570", "rgb(222, 121, 175)"],
  ["0.207843140", "rgb(223, 123, 177)"],
  ["0.211764710", "rgb(224, 126, 179)"],
  ["0.215686280", "rgb(224, 128, 180)"],
  ["0.219607840", "rgb(225, 131, 182)"],
  ["0.223529410", "rgb(226, 133, 184)"],
  ["0.227450980", "rgb(227, 136, 186)"],
  ["0.231372550", "rgb(227, 138, 187)"],
  ["0.235294120", "rgb(228, 141, 189)"],
  ["0.239215690", "rgb(229, 143, 191)"],
  ["0.243137260", "rgb(230, 146, 192)"],
  ["0.247058820", "rgb(230, 148, 194)"],
  ["0.250980400", "rgb(231, 151, 196)"],
  ["0.254901980", "rgb(232, 153, 198)"],
  ["0.258823540", "rgb(233, 156, 199)"],
  ["0.262745100", "rgb(233, 158, 201)"],
  ["0.266666680", "rgb(234, 161, 203)"],
  ["0.270588250", "rgb(235, 163, 205)"],
  ["0.274509820", "rgb(236, 165, 206)"],
  ["0.278431400", "rgb(236, 168, 208)"],
  ["0.282352950", "rgb(237, 170, 210)"],
  ["0.286274520", "rgb(238, 173, 211)"],
  ["0.290196100", "rgb(239, 175, 213)"],
  ["0.294117660", "rgb(239, 178, 215)"],
  ["0.298039230", "rgb(240, 180, 217)"],
  ["0.301960800", "rgb(241, 182, 218)"],
  ["0.305882360", "rgb(241, 184, 219)"],
  ["0.309803930", "rgb(242, 186, 220)"],
  ["0.313725500", "rgb(242, 187, 220)"],
  ["0.317647070", "rgb(243, 189, 221)"],
  ["0.321568640", "rgb(243, 191, 222)"],
  ["0.325490200", "rgb(244, 192, 223)"],
  ["0.329411770", "rgb(244, 194, 224)"],
  ["0.333333340", "rgb(245, 196, 225)"],
  ["0.337254900", "rgb(245, 197, 225)"],
  ["0.341176480", "rgb(245, 199, 226)"],
  ["0.345098050", "rgb(246, 200, 227)"],
  ["0.349019620", "rgb(246, 202, 228)"],
  ["0.352941200", "rgb(247, 204, 229)"],
  ["0.356862750", "rgb(247, 205, 229)"],
  ["0.360784320", "rgb(248, 207, 230)"],
  ["0.364705900", "rgb(248, 209, 231)"],
  ["0.368627460", "rgb(249, 210, 232)"],
  ["0.372549030", "rgb(249, 212, 233)"],
  ["0.376470600", "rgb(250, 214, 234)"],
  ["0.380392160", "rgb(250, 215, 234)"],
  ["0.384313730", "rgb(251, 217, 235)"],
  ["0.388235300", "rgb(251, 219, 236)"],
  ["0.392156870", "rgb(252, 220, 237)"],
  ["0.396078440", "rgb(252, 222, 238)"],
  ["0.400000000", "rgb(253, 224, 239)"],
  ["0.403921570", "rgb(252, 224, 239)"],
  ["0.407843140", "rgb(252, 225, 239)"],
  ["0.411764700", "rgb(252, 226, 239)"],
  ["0.415686280", "rgb(252, 227, 240)"],
  ["0.419607850", "rgb(251, 228, 240)"],
  ["0.423529420", "rgb(251, 229, 240)"],
  ["0.427450980", "rgb(251, 230, 241)"],
  ["0.431372550", "rgb(251, 231, 241)"],
  ["0.435294120", "rgb(250, 232, 241)"],
  ["0.439215700", "rgb(250, 233, 242)"],
  ["0.443137260", "rgb(250, 233, 242)"],
  ["0.447058830", "rgb(250, 234, 242)"],
  ["0.450980400", "rgb(249, 235, 243)"],
  ["0.454901960", "rgb(249, 236, 243)"],
  ["0.458823530", "rgb(249, 237, 243)"],
  ["0.462745100", "rgb(249, 238, 244)"],
  ["0.466666670", "rgb(249, 239, 244)"],
  ["0.470588240", "rgb(248, 240, 244)"],
  ["0.474509800", "rgb(248, 241, 244)"],
  ["0.478431370", "rgb(248, 242, 245)"],
  ["0.482352940", "rgb(248, 242, 245)"],
  ["0.486274500", "rgb(247, 243, 245)"],
  ["0.490196080", "rgb(247, 244, 246)"],
  ["0.494117650", "rgb(247, 245, 246)"],
  ["0.498039220", "rgb(247, 246, 246)"],
  ["0.501960800", "rgb(246, 246, 246)"],
  ["0.505882400", "rgb(246, 246, 244)"],
  ["0.509803950", "rgb(245, 246, 243)"],
  ["0.513725500", "rgb(244, 246, 241)"],
  ["0.517647100", "rgb(244, 246, 240)"],
  ["0.521568660", "rgb(243, 246, 238)"],
  ["0.525490200", "rgb(242, 246, 237)"],
  ["0.529411800", "rgb(242, 246, 235)"],
  ["0.533333360", "rgb(241, 246, 234)"],
  ["0.537254900", "rgb(240, 246, 232)"],
  ["0.541176500", "rgb(240, 246, 230)"],
  ["0.545098070", "rgb(239, 246, 229)"],
  ["0.549019630", "rgb(238, 246, 227)"],
  ["0.552941200", "rgb(238, 245, 226)"],
  ["0.556862800", "rgb(237, 245, 224)"],
  ["0.560784340", "rgb(236, 245, 223)"],
  ["0.564705900", "rgb(236, 245, 221)"],
  ["0.568627500", "rgb(235, 245, 220)"],
  ["0.572549050", "rgb(234, 245, 218)"],
  ["0.576470600", "rgb(234, 245, 217)"],
  ["0.580392200", "rgb(233, 245, 215)"],
  ["0.584313750", "rgb(232, 245, 214)"],
  ["0.588235300", "rgb(232, 245, 212)"],
  ["0.592156900", "rgb(231, 245, 211)"],
  ["0.596078460", "rgb(230, 245, 209)"],
  ["0.600000000", "rgb(230, 245, 208)"],
  ["0.603921600", "rgb(228, 244, 205)"],
  ["0.607843160", "rgb(226, 243, 202)"],
  ["0.611764700", "rgb(224, 242, 199)"],
  ["0.615686300", "rgb(222, 241, 196)"],
  ["0.619607870", "rgb(220, 241, 193)"],
  ["0.623529430", "rgb(219, 240, 190)"],
  ["0.627451000", "rgb(217, 239, 187)"],
  ["0.631372600", "rgb(215, 238, 184)"],
  ["0.635294140", "rgb(213, 237, 181)"],
  ["0.639215700", "rgb(211, 237, 178)"],
  ["0.643137300", "rgb(210, 236, 176)"],
  ["0.647058840", "rgb(208, 235, 173)"],
  ["0.650980400", "rgb(206, 234, 170)"],
  ["0.654902000", "rgb(204, 234, 167)"],
  ["0.658823550", "rgb(202, 233, 164)"],
  ["0.662745100", "rgb(201, 232, 161)"],
  ["0.666666700", "rgb(199, 231, 158)"],
  ["0.670588250", "rgb(197, 230, 155)"],
  ["0.674509800", "rgb(195, 230, 152)"],
  ["0.678431400", "rgb(193, 229, 149)"],
  ["0.682352960", "rgb(192, 228, 147)"],
  ["0.686274500", "rgb(190, 227, 144)"],
  ["0.690196100", "rgb(188, 226, 141)"],
  ["0.694117670", "rgb(186, 226, 138)"],
  ["0.698039230", "rgb(184, 225, 135)"],
  ["0.701960800", "rgb(182, 224, 132)"],
  ["0.705882400", "rgb(180, 222, 129)"],
  ["0.709803940", "rgb(178, 221, 127)"],
  ["0.713725500", "rgb(176, 219, 124)"],
  ["0.717647100", "rgb(173, 218, 121)"],
  ["0.721568640", "rgb(171, 217, 119)"],
  ["0.725490200", "rgb(169, 215, 116)"],
  ["0.729411800", "rgb(167, 214, 113)"],
  ["0.733333350", "rgb(165, 212, 111)"],
  ["0.737254900", "rgb(162, 211, 108)"],
  ["0.741176500", "rgb(160, 209, 105)"],
  ["0.745098050", "rgb(158, 208, 102)"],
  ["0.749019600", "rgb(156, 206, 100)"],
  ["0.752941200", "rgb(153, 205,  97)"],
  ["0.756862760", "rgb(151, 203,  94)"],
  ["0.760784300", "rgb(149, 202,  92)"],
  ["0.764705900", "rgb(147, 201,  89)"],
  ["0.768627460", "rgb(144, 199,  86)"],
  ["0.772549030", "rgb(142, 198,  83)"],
  ["0.776470600", "rgb(140, 196,  81)"],
  ["0.780392170", "rgb(138, 195,  78)"],
  ["0.784313740", "rgb(135, 193,  75)"],
  ["0.788235300", "rgb(133, 192,  73)"],
  ["0.792156900", "rgb(131, 190,  70)"],
  ["0.796078440", "rgb(129, 189,  67)"],
  ["0.800000000", "rgb(127, 188,  65)"],
  ["0.803921600", "rgb(125, 186,  63)"],
  ["0.807843150", "rgb(123, 184,  62)"],
  ["0.811764700", "rgb(121, 183,  61)"],
  ["0.815686300", "rgb(119, 181,  59)"],
  ["0.819607850", "rgb(117, 179,  58)"],
  ["0.823529400", "rgb(115, 178,  57)"],
  ["0.827451000", "rgb(113, 176,  56)"],
  ["0.831372560", "rgb(111, 174,  54)"],
  ["0.835294100", "rgb(109, 173,  53)"],
  ["0.839215700", "rgb(107, 171,  52)"],
  ["0.843137260", "rgb(105, 169,  51)"],
  ["0.847058830", "rgb(103, 168,  49)"],
  ["0.850980400", "rgb(101, 166,  48)"],
  ["0.854901970", "rgb( 99, 164,  47)"],
  ["0.858823540", "rgb( 97, 163,  46)"],
  ["0.862745100", "rgb( 95, 161,  44)"],
  ["0.866666700", "rgb( 93, 160,  43)"],
  ["0.870588240", "rgb( 91, 158,  42)"],
  ["0.874509800", "rgb( 89, 156,  41)"],
  ["0.878431400", "rgb( 87, 155,  39)"],
  ["0.882352950", "rgb( 85, 153,  38)"],
  ["0.886274500", "rgb( 83, 151,  37)"],
  ["0.890196100", "rgb( 81, 150,  36)"],
  ["0.894117650", "rgb( 79, 148,  34)"],
  ["0.898039200", "rgb( 77, 146,  33)"],
  ["0.901960800", "rgb( 76, 145,  32)"],
  ["0.905882360", "rgb( 74, 143,  32)"],
  ["0.909803900", "rgb( 73, 141,  32)"],
  ["0.913725500", "rgb( 71, 139,  31)"],
  ["0.917647060", "rgb( 70, 137,  31)"],
  ["0.921568630", "rgb( 68, 136,  31)"],
  ["0.925490200", "rgb( 67, 134,  30)"],
  ["0.929411770", "rgb( 65, 132,  30)"],
  ["0.933333340", "rgb( 64, 130,  30)"],
  ["0.937254900", "rgb( 62, 128,  30)"],
  ["0.941176500", "rgb( 61, 127,  29)"],
  ["0.945098040", "rgb( 59, 125,  29)"],
  ["0.949019600", "rgb( 58, 123,  29)"],
  ["0.952941200", "rgb( 56, 121,  28)"],
  ["0.956862750", "rgb( 55, 119,  28)"],
  ["0.960784300", "rgb( 53, 118,  28)"],
  ["0.964705900", "rgb( 52, 116,  27)"],
  ["0.968627450", "rgb( 50, 114,  27)"],
  ["0.972549000", "rgb( 49, 112,  27)"],
  ["0.976470600", "rgb( 47, 110,  26)"],
  ["0.980392160", "rgb( 46, 109,  26)"],
  ["0.984313700", "rgb( 44, 107,  26)"],
  ["0.988235300", "rgb( 43, 105,  25)"],
  ["0.992156860", "rgb( 41, 103,  25)"],
  ["0.996078430", "rgb( 40, 101,  25)"],
  ["1.000000000", "rgb( 39, 100,  25)"],
];

// SAWs json data is more complex than needed: reduce it to the necessary
// parts: directories, arrays variables, and strings
const saws_to_plain_json = (saws_json) => {
  let plain = {};
  if (typeof saws_json.directories != "undefined") {
    for (const key in saws_json.directories) {
      plain[key] = saws_to_plain_json(saws_json.directories[key]);
    }
  }
  if (typeof saws_json.variables != "undefined") {
    for (const key in saws_json.variables) {
      plain[key] = saws_json.variables[key].data;
    }
  }
  return plain;
};

// Generate traces from a newly selected finite element
// and make a new plot for them
const generate_traces = (fe) => {
  console.log("Generating element traces");
  const dim = fe.spatial_dimension[0];
  const num_basis = fe.dimension[0];
  const num_comps = fe.number_of_components[0];
  const dual_basis = fe.dual_space;
  const refel = fe.reference_element;
  const num_points = refel.number_of_mesh_points[0];

  let traces = [];
  state.trace_offsets = {refel: 0, dual: 0, shape: 0};
  let mesh_point_start = [num_points, num_points, num_points, num_points];
  const mesh_point_names = ["vertex", "edge", "face", "cell"];

  // We will construct an invisible bounding box to avoid camera movement when restyling
  // we measure the farthest radius from the origin to determine the size of this bounding box
  let radius = 0.0;
  let cone_scale = 0.2;

  // For each mesh point add a trace
  for (let p = 0; p < num_points; p++) {
    const point = refel[p];
    const point_dim = point.dimension[0];
    const num_vertices = point.number_of_vertices[0];

    // mesh points of a dimension are contiguous, find the first one for each type
    mesh_point_start[point_dim] = Math.min(mesh_point_start[point_dim], p);

    // All plotting is 3D, add zero coordinates for unused dimensions
    let x = dim > 0 ? point.coordinates.slice(             0,  num_vertices) : new Float32Array(num_vertices).fill(0.0);
    let y = dim > 1 ? point.coordinates.slice(  num_vertices,2*num_vertices) : new Float32Array(num_vertices).fill(0.0);
    let z = dim > 2 ? point.coordinates.slice(2*num_vertices,3*num_vertices) : new Float32Array(num_vertices).fill(0.0);

    radius = Math.max(radius, Math.max(...x));
    radius = Math.max(radius, -Math.min(...x));
    radius = Math.max(radius, Math.max(...y));
    radius = Math.max(radius, -Math.min(...y));
    radius = Math.max(radius, Math.max(...z));
    radius = Math.max(radius, -Math.min(...z));

    let data = {
      type: point_dim < 2 ? "scatter3d" : "mesh3d", // cells and faces are meshes so 
      x: x,
      y: y,
      z: z,
      hoverinfo: "skip",                           // it would be nice to get hoverinfo in the middle of an edge / face,
                                                   // but that doesn't seem to work, so turn hover off
      marker: { size: 3, color: petsc_logo_blue }, // blue from the logo
      color: petsc_logo_blue,
      visible: (point_dim == 0) ? true : false,    // initially only vertices are visible
    };
    switch (point_dim) {
      case 0:
        data.mode = 'markers';
        break;
      case 1:
        data.mode = 'lines';
        break;
      case 2:
        data.opacity = 0.1;
        { // have to tell plotly which coordinates (not) to use for delaunay triangulation
          const range = (a) => (Math.max(...a) - Math.min(...a));
          let minaxis = 'x';
          let minrange = range(x);
          let yrange = range(y);
          if (yrange < minrange) {
            minrange = yrange;
            minaxis = 'y';
          }
          let zrange = range(z);
          if (zrange < minrange) {
            minrange = zrange;
            minaxis = 'z';
          }
          data.delaunayaxis = minaxis; // axis with minimum difference is the skipped coordinates
        }
        break;
      case 3:
        data.opacity = 0.1;
        data.alphahull = 0; // use convex hull algorithm
        break;
      default:
        data.mode = 'none';
        break;
    }
    traces.push(data);
  }
  state.trace_offsets.dual = num_points;

  // plot nodal dofs
  // TODO: handle modal

  // compute the maximum dof norm for scaling
  let max_dof_norm = 0.0;
  for (let b = 0; b < num_basis; b++) {
    const functional = dual_basis[b];
    const num_nodes = functional.number_of_nodes;
    const weights = functional.weights;
    for (let n = 0; n < num_nodes; n++) {
      const node_weights = weights.slice(n*num_comps, (n+1)*num_comps);
      let norm = 0.0;
      for (let c = 0; c < num_comps; c++) {
        let w = node_weights[c];
        norm += w*w;
      }
      norm = Math.sqrt(norm);
      max_dof_norm = Math.max(max_dof_norm, norm);
    }
  }
  console.log("Maximum dof norm " + max_dof_norm);


  for (let b = 0; b < num_basis; b++) {
    const functional = dual_basis[b];
    const num_nodes = functional.number_of_nodes[0];
    const nodes = functional.nodes;
    const weights = functional.weights;
    const mesh_point = functional.mesh_point[0];
    const mesh_point_dim = refel[mesh_point].dimension[0];

    let x = dim > 0 ? nodes.slice(          0,  num_nodes) : new Float32Array(num_nodes).fill(0.0);
    let y = dim > 1 ? nodes.slice(  num_nodes,2*num_nodes) : new Float32Array(num_nodes).fill(0.0);
    let z = dim > 2 ? nodes.slice(2*num_nodes,3*num_nodes) : new Float32Array(num_nodes).fill(0.0);
    let this_radius = 0.0
    this_radius = Math.max(this_radius, Math.max(...x));
    this_radius = Math.max(this_radius, -Math.min(...x));
    this_radius = Math.max(this_radius, Math.max(...y));
    this_radius = Math.max(this_radius, -Math.min(...y));
    this_radius = Math.max(this_radius, Math.max(...z));
    this_radius = Math.max(this_radius, -Math.min(...z));
    let u = num_comps > 0 ? weights.slice(          0,  num_nodes) : new Float32Array(num_nodes).fill(0.0);
    let v = num_comps > 1 ? weights.slice(  num_nodes,2*num_nodes) : new Float32Array(num_nodes).fill(0.0);
    let w = num_comps > 2 ? weights.slice(2*num_nodes,3*num_nodes) : new Float32Array(num_nodes).fill(0.0);
    let colors = new Float32Array(num_nodes).fill(0.0);
    let this_max_norm = 0.0;
    for (let n = 0; n < num_nodes; n++) {
      const node_weights = weights.slice(n*num_comps, (n+1)*num_comps);
      let norm = 0.0;
      for (let c = 0; c < num_comps; c++) {
        let w = node_weights[c];
        norm += w*w;
      }
      norm = Math.sqrt(norm);
      this_max_norm = Math.max(this_max_norm, norm);
      colors[n] = norm / max_dof_norm;
    }
    radius = Math.max(radius, this_radius + cone_scale * ((num_comps > 1) ? this_max_norm : 0.0));
    let data = {
      type: (num_comps == 1) ? "scatter3d" : "cone",
      x: x,
      y: y,
      z: z,
      u: u,
      v: v,
      w: w,
      name: "dof " + b,
      text: "mesh point " + mesh_point + " (" + mesh_point_names[mesh_point_dim] + " " + (mesh_point - mesh_point_start[mesh_point_dim]) + ")",
      hoverinfo: "text+name" + ((dim > 0) ? "+x" : "") + ((dim > 1) ? "+y" : "") + ((dim > 2) ? "+z" : "") + ((num_comps > 1) ? "+norm+u+v" : "") + ((num_comps > 2) ? "+w" : ""),
      visible: false,
      marker: {
        size: 3,
        cmin: -1,
        cmax: 1,
        color: colors,
        colorscale: PiYG,
      },
      cmin: -1,
      cmax: 1,
      color: colors,
      colorscale: PiYG,
      showscale: false,
      anchor: "tail",
      sizemode: "relative",
      sizeref: cone_scale,
    };
    traces.push(data);
  }
  state.trace_offsets.shape = num_points + num_basis;
  const layout = {
    showlegend: false,
    title: fe.name,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      // aspectratio: {x: 1, y: 1, z: 1},
      // aspectmode: "cube",
      xaxis: { showgrid: false, visible: (dim > 0), },
      yaxis: { showgrid: false, visible: (dim > 1), },
      zaxis: { showgrid: false, visible: (dim > 2), },
      camera: {
        eye: (dim < 3) ? {x: 0, y: 0, z: 1} : {x: 1.25, y: 1.25, z: 1.25},
        up: {
          x: 0, 
          y: (dim < 3) ? 1 : 0,
          z: (dim < 3) ? 0 : 1,
        },
        projection: {type : (dim < 3) ? "orthographic" : "perspective" },
      },
      dragmode: (dim < 3) ? "pan" : "orbit",
    },
  };
  // add the bounding box
  traces.push({
    type: 'mesh3d',
    opacity: 0.0,
    hoverinfo: "skip",
    x: [-radius, +radius, -radius, +radius, -radius, -radius, +radius, -radius],
    y: [-radius, -radius, +radius, +radius, -radius, -radius, +radius, +radius],
    z: [-radius, -radius, -radius, -radius, +radius, +radius, +radius, +radius],
  });
  const plot_div = document.getElementById('feplotly_plot');
  Plotly.newPlot(plot_div, traces, layout);
};

const replot = (fe) => {
  console.log("Updating plotly plot");

  let visible = { visible: true };
  let visible_vertex = { opacity: 1.0 };
  let visible_list = [];
  let visible_vertex_list = [];
  let invisible = { visible: false };
  let invisible_vertex = { opacity: 0.1 };
  let invisible_list = [];
  let invisible_vertex_list = [];

  const selected_dof = document.getElementById("feplotly_select_dof").value;
  const selected_mesh_point = (selected_dof == 'all') ? 'all' : fe.dual_space[selected_dof].mesh_point;

  const refel = fe.reference_element;
  const num_points = refel.number_of_mesh_points[0];
  const plot_refel = document.getElementById("feplotly_plot_refel").checked;
  for (let p = 0; p < num_points; p++) {
    if (plot_refel && (selected_mesh_point == 'all' || selected_mesh_point == p)) {
      if (refel[p].dimension[0] == 0) {
        visible_vertex_list.push(p + state.trace_offsets.refel);
      }
      else {
        visible_list.push(p + state.trace_offsets.refel);
      }
    }
    else {
      if (refel[p].dimension[0] == 0) {
        invisible_vertex_list.push(p + state.trace_offsets.refel);
      }
      else {
        invisible_list.push(p + state.trace_offsets.refel);
      }
    }
  }

  const num_basis = fe.dimension[0];
  const plot_dual = document.getElementById("feplotly_plot_dual").checked;
  for (let b = 0; b < num_basis; b++) {
    if (plot_dual && (selected_dof == 'all' || selected_dof == b)) {
      visible_list.push(b + state.trace_offsets.dual);
    }
    else {
      invisible_list.push(b + state.trace_offsets.dual);
    }
  }

  const plot_div = document.getElementById('feplotly_plot');
  console.log(invisible_list);
  if (visible_vertex_list.length > 0) {
    Plotly.restyle(plot_div, visible_vertex, visible_vertex_list);
  }
  if (invisible_vertex_list.length > 0) {
    Plotly.restyle(plot_div, invisible_vertex, invisible_vertex_list);
  }
  if (visible_list.length > 0) {
    Plotly.restyle(plot_div, visible, visible_list);
  }
  if (invisible_list.length > 0) {
    Plotly.restyle(plot_div, invisible, invisible_list);
  }
}

const refresh_properties = (fe) => {
  console.log("Loading element properties");
  document.getElementById("fe_name").innerHTML = fe.name;
  if (fe.options_prefix != "") {
    document.getElementById("fe_prefix").innerHTML = '<code>-' + fe.options_prefix + '</code>';
  }
  else {
    document.getElementById("fe_prefix").innerHTML = "&nbsp;";
  }
  document.getElementById("fe_spatial_dimension").innerHTML = fe.spatial_dimension;

  document.getElementById("fe_dimension").innerHTML = fe.dimension;
  let dof_options = '<option>all</option>';
  for (let b = 0; b < fe.dimension; b++) {
    dof_options += '<option>' + b + '</option>';
  }
  document.getElementById("feplotly_select_dof").innerHTML = dof_options;
  document.getElementById("fe_refel").innerHTML = fe.reference_element[0].polytope;
  document.getElementById("fe_min_degree").innerHTML = fe.basis_space.minimum_degree;
  document.getElementById("fe_max_degree").innerHTML = fe.basis_space.maximum_degree;
  document.getElementById("fe_variance").innerHTML = fe.dual_space.variance;
  document.getElementById("fe_continuity").innerHTML = fe.dual_space.continuity;
  MathJax.typeset();
};

const feviewer_refresh = () => {
  let selected_element = document.querySelector("input[name=feviewer_element_select]:checked").value;
  if (state.selected_element != selected_element) {
    console.log("Switch to element " + selected_element);
    state.selected_element = selected_element;
    refresh_properties(fe_data[selected_element]);
    generate_traces(fe_data[selected_element]);
  }
  replot(fe_data[selected_element]);
};

const hangup_json = {
  directories: {
    SAWs_ROOT_DIRECTORY: {
      variables: {
        __Block: {
          data: ["false"],
          dtype: "SAWs_BOOLEAN",
          length: 1,
          mtype: "SAWs_WRITE"
        }
      }
    }
  }
};

const hangup_options = {
  method: "POST",
  body: "input=" + JSON.stringify(hangup_json),
  headers: {
    "Content-Type": "application/json"
  }
};


const feviewer_init = (fe_saws_json) => {
  try {
    fe_data = saws_to_plain_json(fe_saws_json).FE;
    if (fe_data.hasOwnProperty("completed")) {
      fetch("/SAWs/*", hangup_options);
    }
    const number_of_elements = fe_data.number_of_elements[0];
    console.log("Loading " + number_of_elements + " finite elements");
    const tabs_div = document.getElementById('feviewer_radio_tabs');
    for (let i = 0; i < number_of_elements; i++) {
      let radio_tab = document.createElement("INPUT");
      radio_tab.id = 'feviewer_radio_tab' + i;
      radio_tab.type = 'radio';
      radio_tab.classList.add('feviewer_radio_tab');
      radio_tab.value = i;
      radio_tab.name = 'feviewer_element_select';
      radio_tab.checked = (i == 0);
      radio_tab.addEventListener("click", feviewer_refresh);
      let tab_label = document.createElement("LABEL");
      tab_label.id = 'feviewer_radio_tab_label' + i;
      tab_label.htmlFor = radio_tab.id;
      if (fe_data[i].options_prefix != "") {
        tab_label.innerHTML = '<h2>' + fe_data[i].name + ' (<code>-' + fe_data[i].options_prefix + '</code>)</h2>';
      }
      else {
        tab_label.innerHTML = '<h2>' + fe_data[i].name + '</h2>';
      }
      tabs_div.appendChild(radio_tab);
      tabs_div.appendChild(tab_label);
    }
    document.getElementById("feplotly_select_dof").addEventListener("change", feviewer_refresh);
    document.getElementById("feplotly_plot_refel").addEventListener("change", feviewer_refresh);
    document.getElementById("feplotly_plot_dual").addEventListener("change", feviewer_refresh);
    window.addEventListener("resize", feviewer_refresh());
    feviewer_refresh();
  }
  catch (e) {
    console.log(e);
    document.getElementById('feviewer_status').innerHTML = "FE error, see console log";
  }
};

const no_fe_network_error = (err) => {
  console.log(err);
  document.getElementById('feviewer_status').innerHTML = `
        <div style="error">
          <h2>Connection error</h2>

          <p>The SAWs server could not be reached.</p>

          <p>If you are running PETSc locally, do not try to load this webpage
             from <code>file://.../fe/</code>, use <code>http://localhost:&lt;port&gt;/fe/</code>.
             The default port is 8080, but may have been set with the command line
             option <code>-saws_port YYYY</code>.</p>
        </div>
     `;
};

const no_fe_json_error = (err) => {
  console.log(err);
  document.getElementById('feviewer_status').innerHTML = `
        <div style="error">
          <h2>Loading error</h2>

          <p>The finite element data could not be loaded.</p>

          <p>Your program may not have viewed any finite elements with the appropriate viewer.
             If your finite elements are created by <code>PetscFECreateDefault()</code> or
             <code>PetscFECreateLagrange()</code>, rerun your program with
             <code>-&lt;prefix&gt;petscfe_view saws:</code>,
             where <code>&lt;prefix&gt;</code> is the options prefix for your finite element.</p>
        </div>
     `;
};

window.addEventListener("load", () => {
  fetch('/SAWs/PETSc/FE/')
    .then(result => {
      return result.json()
        .then(data => feviewer_init(data))
        .catch(err => no_fe_json_error(err));
    })
    .catch(err => no_fe_network_error(err));
});


