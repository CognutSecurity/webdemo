# -*- coding: utf-8 -*-

cleaner = {
    "interval": 60,
    "old": 24 * 60 * 60
}

camera = {
    "index": -1,
    "count": 50,
    "width": 640,
    "height": 480,
    "interval": 15,
    "folder": "_images"
}

server = {
    "host": "0.0.0.0",
    "port": 81,
    "html": """<html>
    <head>
        <meta charset="utf-8">
        <title>WebCameraServer</title>
        <style type="text/css">
            body {
                font: 16px sans-serif;
                margin: 0;
                padding: 0;
            }

            #image {
                background: #263238;
                height: 100vh;
                min-height: %dpx;
                min-width: %dpx;
                text-align: center;
                width: 100vw;
            }

            #image:before {
                content: " ";
                display: inline-block;
                height: 100vh;
                vertical-align: middle;
            }

            #image img {
                display: inline-block;
                vertical-align: middle;
            }

            #images {
                bottom: 10px;
                position: fixed;
                right: 10px;
                text-align: center;
                width: 150px;
            }

            #images a,
            #images a:active {
                color: #CFD8DC;
                text-decoration: none;
            }

            #images a:focus,
            #images a:hover {
                color: #B0BEC5;
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div id="image"></div>
        <div id="images">
            <a href="/images" target="_blank">Download images</a>
        </div>
        <script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
        <script type="text/javascript">
            (function ($) {
                "use strict";

                $(function () {
                    function getImage() {
                        $.ajax({
                            "url": "/image"
                        }).done(function (data) {
                            $("#image").html(data);
                        });
                    }

                    getImage();

                    setInterval(getImage, 1000);
                });
            }(jQuery));
        </script>
    </body>
</html>""" % (camera['height'], camera['width'])
}