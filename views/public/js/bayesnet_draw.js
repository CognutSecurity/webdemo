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
         method: "POST",
         dataType: "json"
      }).done(function(response) {
         // body...
         //  alert(response.node_names);
         var elements = [];
         for (var node in response.node_names) {
            elements.push({
               data: {
                  "id": response.node_names[node]
               }
            });
         }
         for (var e in response.edges) {
            edge = response.edges[e];
            elements.push({
               data: {
                  "id": edge.id.toFixed(2),
                  "source": edge.source,
                  "target": edge.target
               }
            });
         }
         init_graph(elements);
      });
   });
});

function init_graph(elements) {
   // Initialize the graph
   var g = cytoscape({
      container: $('#graph_canvas'),
      elements: elements,
      style: [ // the stylesheet for the graph
         {
            selector: 'node',
            style: {
               'background-color': '#177',
               'label': 'data(id)',
               'width': 20,
               'height': 20
            }
         },

         {
            selector: 'edge',
            style: {
               'width': 3,
               'curve-style': 'bezier',
               'line-color': '#aaa',
               'target-arrow-color': '#aaa',
               'target-arrow-shape': 'triangle',
               'label': 'data(id)'
            }
         }
      ],

      layout: {
         name: 'circle',
         fit: true
      },
      // initial viewport state:
      zoom: 1,
      pan: {
         x: 0,
         y: 0
      },

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
