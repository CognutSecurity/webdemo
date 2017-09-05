$(document).ready(function() {
  
  // init code snippet editor
  var codeMirror = CodeMirror.fromTextArea(document.getElementById('editor'), {
    lineNumbers: true,
    matchBrackets: true,
    mode: "text/x-java"
  });

  // analyze snippet in editor
  $("#analyze-security").click(function(e) {
    analyzeCodeInEditor();
  });

  // init .csv file and analyze snippets
  $("#btn-explore").click(function(e) {
    e.stopPropagation();
    e.preventDefault();

    if ($("#upload-input").val() == '') {
      alert("No file selected!");
      e.preventDefault();
      return;
    }
    
    var form = new FormData();
    form.append('upload_input', $("#upload-input").prop("files")[0], $("#upload-input").name);
    initSnippetCards(form);
  });
  
  $("#next-snippet").click(function(e) {
    var ps  = postSnippet(),
        pss = ps.then(function(response) {
          var snippet = response.snippet;
          setSnippet('#pretty-input-snippet','#input-snippet-card', snippet);
          return putSimilarSnippet(snippet);
        }),
        gss = pss.then(function() {
          return getSimilarSnippet(); 
        }),
        psec = gss.then(function(response) {
          var similarSnippet = response.snippet;
          setSnippet('#pretty-input-similar-snippet', '#similar-snippet-card', similarSnippet);
        });
  });

  $("#next-similar-snippet").click(function(e) {
    postSimilarSnippet().done(function(response) {
      var similarSnippet = response.snippet;
      setSnippet('#pretty-input-similar-snippet', '#similar-snippet-card', similarSnippet);
    });
  });
});

function analyzeCodeInEditor() {
  var code = $('#editor').val();
    
  $.ajax({
    url: "/stacksecure/security",
    data: {'snippet' : code},
    method: "POST"
  }).done(function(response) {
    setSnippetCard('#code-snippet-card', response);
  });
}

function initSnippetCards(form) {
  var idb = initDatabase(form),
      ps  = idb.then(function() {
        return putSnippet();
      }),
      gs  = ps.then(function() {
        return getSnippet();
      }),
      pss = gs.then(function(response) {
        var snippet = response.snippet;
        setSnippet('#pretty-input-snippet','#input-snippet-card', snippet);
        return putSimilarSnippet(snippet);
      }),
      gss = pss.then(function() {
        return getSimilarSnippet();
      }),
      psec = gss.then(function(response) {
        var similarSnippet = response.snippet;
        setSnippet('#pretty-input-similar-snippet', '#similar-snippet-card', similarSnippet);
      });
}

function setSnippet(pretty, card, snippet) {
  setPrettyPrint(pretty, snippet, 'java');
  postSecurity(snippet).done(function(response) {
    setSnippetCard(card, response);
  });
}

function setSnippetCard(card, security) {
  if (security == "1") {
    $(card).removeClass("card-default card-success");
    $(card).addClass("card-danger");
  } else if (security == "-1") {
    $(card).removeClass("card-default card-danger");
    $(card).addClass("card-success");
  }
}

function setPrettyPrint(card, rS, lang) {
  $(card)
    .html(prettyPrintOne(rS, lang))
    .addClass('prettyprint lang-java prettyprinted');
}

function initDatabase(form) {
  return $.ajax({
    url: "/stacksecure/snippetdatabasehelper",
    processData: false,
    contentType: false,
    data: form,
    method: "PUT"
  });
}

function getSnippet() {
  return $.ajax({
    url: "/stacksecure/snippet",
    method: "GET"
  });
}

function postSnippet() {
  return $.ajax({
    url: "/stacksecure/snippet",
    method: "POST"
  });
}

function putSnippet() {
  return $.ajax({
    url: "/stacksecure/snippet",
    method: "PUT"
  });
}

function getSimilarSnippet() {
  return $.ajax({
    url: "/stacksecure/similarsnippet",
    method: "GET"
  });
}

function postSimilarSnippet() {
  return $.ajax({
    url: "/stacksecure/similarsnippet",
    method: "POST",
  });
}

function putSimilarSnippet(snippet) {
  return $.ajax({
    url: "/stacksecure/similarsnippet",
    method: "PUT",
    data: {'snippet' : snippet }
  });
}

function postSecurity(snippet) {
  return $.ajax({
    url: "/stacksecure/security",
    method: "POST",
    data: {'snippet' : snippet}
  });
}
