$(document).ready(function() {
   var elements = [];
   var elems = $('#graph_canvas').data('elements').replace(/'/g, '"').slice(1,-1).split(', ');
   console.log(elems);
   for (e in elems) {
     elements.push(JSON.parse(elems[e].trim()));
   }
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
});
