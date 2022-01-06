function update_concept_animation_figure() {
  var filename = "./images/concept_animation_figure/" + document.getElementById("concept_animation_binding_domain").value; 
  var image = document.getElementById("concept_animation");
  image.src = filename;
}

function update_sensor_mechanism_comparison_figure() {
  var filename = "./images/sensor_mechanism_comparison_figure/" + document.getElementById("sensor_mechanism_comparison_figure_choice").value; 
  var image = document.getElementById("sensor_mechanism_comparison");
  image.src = filename;
}

function update_autofluorescence_overlay_figure() {
  var filename = "./images/autofluorescence_figure/" + document.getElementById("autofluorescence_overlay_figure_image_type").value; 
  var image = document.getElementById("autofluorescence_figure_overlay");
  image.src = filename;
}

function update_ph_sensing_data_figure() {
  var interval = document.getElementById("ph_sensing_data_figure_relaxation_interval").value;
  var type = document.getElementById("ph_sensing_data_figure_image_type").value; 
  if (type.endsWith(".gif")) {
    var desc = "animation";
    var pref = "1_";}
  else {
    var desc = "frame";
    var pref = ""}
  var filename = "./images/ph_sensing_figure/" + pref + "data_" + desc + "_" + interval + type;
  var image = document.getElementById("ph_sensing_figure_data_animation");
  image.src = filename;
}

function update_ph_sensing_colormap_figure() {
  var measurement_type = document.getElementById("ph_sensing_figure_measurement_type").value; 
  var colormap_type = document.getElementById("ph_sensing_figure_colormap_type").value; 
  var interval = document.getElementById("ph_sensing_colormap_figure_relaxation_interval").value;
  var filename = "./images/ph_sensing_figure/" + measurement_type + colormap_type + interval + ".png";
  var image = document.getElementById("ph_sensing_figure_colormap");
  image.src = filename;
}

function update_mouse_mito_bam_figure() {
  var comparison = document.getElementById("mouse_mito_bam_comparison").value; 
  var filename = "./images/mouse_mito_bam_figure/" + comparison + ".gif";
  var image = document.getElementById("mouse_mito_bam_gif");
  image.src = filename;
}

function update_robustness_quantitation_figure() {
  var filename = "./images/robustness_quantitation_figure/" + document.getElementById("robustness_animation_").value; 
  var image = document.getElementById("robustness_animation");
  image.src = filename;
  }  

function update_buffer_sensitivity_figure() {
  var filename = "./images/buffer_sensitivity_figure/" + document.getElementById("buffer_sensitivity_").value; 
  var image = document.getElementById("buffer_sensitivity");
  image.src = filename;
  }  

  function update_temperature_sensitivity_figure() {
  var filename = "./images/temperature_sensitivity_figure/" + document.getElementById("temperature_sensitivity_").value; 
  var image = document.getElementById("temperature_sensitivity");
  image.src = filename;
  } 

    function update_yeast_ph_figure() {
  var filename = "./images/yeast/" + document.getElementById("yeast_ph_").value; 
  var image = document.getElementById("yeast_ph");
  image.src = filename;
  } 
