$(document).ready(function() {

   // add ajax on btn_explore file upload button
   $("#bnt_explore").click(function(e) {
      // body...
      e.stopPropagation();
      e.preventDefault();

      if ($("#upload_input").val() == '') {
         alert("No file selected!");
         e.preventDefault();
         return;
      }

//      var uploaded_file = $("#upload_input").prop("files")[0];
       var form = new FormData();
       form.append('upload_input', $("#upload_input").prop("files")[0], $("#upload_input").name);
      // ajax process uploaded file
       $.ajax({
          url: "/bayesnet/draw",
          data: form,
          cache: false,
          contentType: false,
          processData: false,
          type: 'POST'}).done(function(r) {
             // body...
//             alert(JSON.parse(r))
//             init_graph(r);
            var elems = r;
            init_graph(elems);
          });
//      $.post("/bayesnet/draw", {"data": uploaded_file}).done(function(res) {
//         // body...
//         alert(res);
//      });
   });
});

function init_graph (elements) {
   // body...
   var g = cytoscape({
      container: $('#graph_canvas'),
      elements: elements,
      style: [ // the stylesheet for the graph
         {
            selector: 'node',
            style: {
               'background-color': '#177',
               'label': 'data(id)',
               'width': 10,
               'height': 10
            }
         },

         {
            selector: 'edge',
            style: {
               'width': 2,
               'line-color': '#ccc',
               'target-arrow-color': '#ccc',
               'target-arrow-shape': 'triangle'
            }
         }
      ],
      layout: {
         name: 'grid',
         rows: 2
      },
      // initial viewport state:
     zoom: 1,
     pan: { x: 0, y: 0 },

     // interaction options:
     minZoom: 1e-50,
     maxZoom: 1e50,
     zoomingEnabled: true,
     userZoomingEnabled: true,
     panningEnabled: true,
     userPanningEnabled: true,
     boxSelectionEnabled: false,
     selectionType: 'single',
     touchTapThreshold: 8,
     desktopTapThreshold: 4,
     autolock: false,
     autoungrabify: false,
     autounselectify: false,

     // rendering options:
     headless: false,
     styleEnabled: true,
     hideEdgesOnViewport: false,
     hideLabelsOnViewport: false,
     textureOnViewport: false,
     motionBlur: false,
     motionBlurOpacity: 0.2,
     wheelSensitivity: 1,
     pixelRatio: 'auto'
   });
}
