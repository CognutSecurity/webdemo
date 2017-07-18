$(document).ready(function() {
   var ci_coef = "{{ ci_coef }}"
   var adjmat = "{{ adjmat }}"
   console.log(ci_coef);
   console.log(adjmat);
   var g = cytoscape({
      container: $('#graph_canvas'),

      elements: [ // list of graph elements to start with
         { // node a
            data: {
               id: 'a'
            }
         },
         { // node b
            data: {
               id: 'b'
            }
         },
         { // node b
            data: {
               id: 'c'
            }
         },
         { // node b
            data: {
               id: 'd'
            }
         },
         { // edge ab
            data: {
               id: 'ab',
               source: 'a',
               target: 'b'
            }
         },
         { // edge ab
            data: {
               id: 'ac',
               source: 'a',
               target: 'c'
            }
         },
         { // edge ab
            data: {
               id: 'bc',
               source: 'b',
               target: 'c'
            }
         },
         {
            data: {
               id: 'cd',
               source: 'c',
               target: 'd'
            }
         }
      ],

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
