function updateTags(thres) {
    // use this function to hide tags less than certain probability
    $(".badge").each(function() {
        if (parseFloat($(this).html().trim(), 10) <= thres) {
            $(this).parent().hide();
        } else {
            $(this).parent().show();
        }
    });
}

$.expr[':'].containsIgnoreCase = function(n, i, m) {
    return jQuery(n).text().toUpperCase().indexOf(m[3].toUpperCase()) >= 0;
};


$(document).ready(function() {
   // var elem = document.querySelector('.js-range');
   // var init = new Powerange(elem, {min: 0, max: 1, step: 0.02, decimal: true, hideRange: true});
   // we plot
   $('div#we_modal').hide();
   $('#we_modal_btn').on("click", function(e) {
     $("div#we_modal").modal('toggle');
   });

   // search similar words and highlight them
   $("#search_similar_btn").click(function(e) {
     $.post("/getSimilarWords", {
             "word": $("#watch_word_input").val()
         })
         .done(function(s) {
             $("span.word_tag").removeClass("label label-info");
             if ($.isEmptyObject(s)) {
                $("#alert_msg").html("<b>Not found, not in vocabulary.</b>");
             } else {
                $("#alert_msg").html("<b>Similar words found: </b>");
                for (var w in s) {
                  //   $("span.word_tag:containsIgnoreCase('" + w + "')").toggleClass("highlighted");
                    $("span.word_tag").filter(function() {
                       return $(this).text().trim().toLowerCase() === s[w];
                    }).addClass("label label-info");
                    $("#alert_msg").append("<span class='label label-info' style='margin-left: 5px;'>" + s[w] + "</span>");
                }
             }
             $("div.alert").show(200);
         });
     e.preventDefault();
   });

   // hiden alert info when click close
   $("div.alert button.close").on("click", function(e) {
      $("div.alert").hide();
      $("span.word_tag").removeClass("label label-info");
   });

});
// $("div.alert").hide();
// $("#info_btn").click(function(e) {
//    $.get("/getWordVec")
//    .done(function(s) {
//       $("div.alert").html(s);
//       $("div.alert").show();
//    });
//    e.preventDefault();
// });

// $(".sentence-panel .panel-body").on({
//       mouseenter: function(e) {
//          var sent = $(this).html().replace(/\n/g, "<br>");
//          var associate_mail = $(this).parentsUntil(".panel-success").last();
//          var mail_body = associate_mail.find("p").html();
//          var start_idx = mail_body.indexOf(sent);
//          var end_idx = start_idx + sent.length;
//          var new_mail_body = mail_body.substr(0, start_idx) + "<mark>" +
//                      mail_body.substr(start_idx, sent.length) + "</mark>" +
//                      mail_body.substr(end_idx, mail_body.length - end_idx);
//          associate_mail.find("p").html(new_mail_body);
//       },
//       mouseleave: function(e) {
//          var associate_mail = $(this).parentsUntil(".panel-success").last();
//          var mail_body = associate_mail.find("p").html();
//          var new_mail_body = mail_body.replace("<mark>", "").replace("</mark>", "");
//          associate_mail.find("p").html(new_mail_body);
//       }
//    });
