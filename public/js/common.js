$(document).ready(function() {
   // add nav links handlers
   addTopNavLinksHandlers();
});

function addTopNavLinksHandlers() {
   // automatically set active link
   $("#sidebar-menu > li").click(function(e) {
      $("#sidebar-menu > li").removeClass("active");
      e.target.parentElement.classList.add("active");
   });
}
