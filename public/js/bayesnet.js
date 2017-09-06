$(document).ready(function() {
   // add handler to choose data list
   addAnalyzeHandlers();
   addChooseDataHandlers();
   // $(".form-select-file .table-datalist").hide();
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
               'width': 15,
               'height': 15
            }
         },

         {
            selector: 'edge',
            style: {
               'width': 2,
               'curve-style': 'bezier',
               'line-color': '#D7DDE4',
               'target-arrow-color': '#D7DDE4',
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

   var dontblur = false;
   $(".form-select-file .table-datalist").mouseover(function(e) {
      dontblur = true;
   });

   $(".form-select-file .table-datalist").mouseleave(function(e) {
      dontblur = false;
   });

   $("#input_data").focus( function() {
      // display data list
      $(".form-select-file .table-datalist").css('visibility', 'visible');
   });

   $("#input_data").blur( function() {
      // hide data list
      if (!dontblur) {
         $(".form-select-file .table-datalist").css('visibility', 'hidden');
      }
   });

   $(".btn-choose-data").click( function(e) {
      $("#input_data").val($(e.target).data('inputname'));
      $(".form-select-file .table-datalist").css('visibility', 'hidden');
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
            $("#graph_canvas").html("<img id='img-loading' src='/imgs/simple_loading.gif'>");
            socket.send(JSON.stringify({'cmd': 'train', 'formData': form}))
         }
      };

      socket.onmessage = function(e) {
         resp = JSON.parse(e.data);
         if (resp.hasOwnProperty('stats') && resp.stats == 100) {
            $("#graph_canvas").empty();
            $("#sample-dist-canvas").empty();
            // plot nodes DAG
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

            // plot sample distribution 
            var samples = {
                x: resp.samples2d['x'],
                y: resp.samples2d['y'],
                mode: 'markers',
                type: 'scatter',
                marker: { color: '#177' }
              };
            var data = [samples];
            Plotly.newPlot("sample_dist_canvas", data);
            // only close socket if it sucesses
            socket.close(1000);
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
