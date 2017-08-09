$(document).ready(function() {
   // add handler to choose data list
   addAnalyzeHandlers();
   addChooseDataHandlers();
});

function initGraph(elements) {
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
               'width': 2.5,
               'curve-style': 'bezier',
               'line-color': '#aaa',
               'target-arrow-color': '#aaa',
               'target-arrow-shape': 'triangle',
               'label': 'data(id)',
               'text-rotation': 'autorotate'
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

function addChooseDataHandlers() {
   $("#input_data").focus( function() {
      // display data list
      $(".form-select-file .table-datalist").css("display", "block");
   });

   $("#input_data").blur( function() {
      // hide data list
      $(".form-select-file .table-datalist").css("display", "none");
   });

   $(".btn-choose-data").click( function(e) {
      // FIXME: did not work yet
      $("#input_data").val($(e.target).data('inputname'));
   } );

   $("#choosefile-input").click(function() {
      // trigger click upload button
      $("input[name='upload_input']").trigger('click');
   });

   $("input[name='upload_input']").change(function(e) {
      // TODO: ask before upload
      var filename = $("input[name='upload_input']").val().split("\\");
      if (filename) {
         $("#input_data").val(filename[filename.length-1]);
         // TODO: upload file
         var form = new FormData();
         form.append('upload_input', $("input[name='upload_input']").prop("files")[0]);
         $.ajax({
            url: "/inventory/upload",
            data: form,
            cache: false,
            contentType: false,
            processData: false,
            method: "POST",
            dataType: "json"
         }).done(function(resp) {
            // show info
            alert(resp['info']);
         });
      }
   });
}

function addAnalyzeHandlers() {
   // play btn clicked
   $("#play-pause-btn").click( function(e) {
      var socket = new WebSocket("ws://"+window.location.hostname+":8080/bayesnet/ws")

      socket.onopen = function(e) {
         var dataName = $("#input_data").val();
         if (!dataName) {
            alert('No data selected!');
            return;
         } else {
            var form = {};
            form['datafile_name'] = dataName;
            form['alpha'] = $("#alphas").val();
            form['method'] = $("#methods").val();
            form['penalty'] = $("#penalties").val();
            form['bin'] = $("#bins").val();
            form['pval'] = $("#pvals").val();
            form['ssample'] = $("#ssamples").val();
            $("#graph_canvas").html("<img src='/imgs/simple_loading.gif' style='display: block; margin: auto;'>");
            socket.send(JSON.stringify({'cmd': 'train', 'formData': form}))
         }
      };

      socket.onmessage = function(e) {
         resp = JSON.parse(e.data);
         if (resp.hasOwnProperty('stats') && resp.stats == 100) {
            $("#graph_canvas").empty();
            var elements = [];
            for (var node in resp.node_names) {
               elements.push({
                  data: {
                     "id": resp.node_names[node]
                  }
               });
            }
            for (var e in resp.edges) {
               edge = resp.edges[e];
               elements.push({
                  data: {
                     "id": edge.id.toFixed(2),
                     "source": edge.source,
                     "target": edge.target
                  }
               });
            }
            initGraph(elements);
            $("#debug-canvas p").empty();
         } else {
            if (resp.hasOwnProperty('stats') && resp.stats == 99) {
               // 99: in progress
               $("#debug-canvas p").html(resp.info);
            } else if (resp.hasOwnProperty('stats') && resp.stats == 101) {
               // 101: errors
               $("#graph_canvas").empty();
               $("#graph_canvas").html("<p class=text-danger>" + resp.info);
            }
         }
      };
   });
}
