$(document).ready(function () {
  
  // init code snippet editor
  var codeMirror = CodeMirror.fromTextArea(document.getElementById('editor'), {
    lineNumbers: true,
    matchBrackets: true,
    mode: "text/x-java"
  });

  var init_snippet = 
  `// Insert security-related Java code snippet
  SecretKeySpec getKey() {
  final pass = "47e7717f0f37ee72cb226278279aebef".getBytes("UTF-8");
  final sha = MessageDigest.getInstance("SHA-256");

  def key = sha.digest(pass);
  // use only first 128 bit (16 bytes). By default Java only supports AES 128 bit key sizes for encryption.
  // Updated jvm policies are required for 256 bit.
  key = Arrays.copyOf(key, 16);
  return new SecretKeySpec(key, AES);
}`;
  
  codeMirror.setValue(init_snippet);

  // analyze snippet in editor
  $("#analyze-security").click(function(e) {
    analyzeCodeInEditor(codeMirror);
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

function analyzeCodeInEditor(codeMirror) {
  // var code = $('#editor').val();
  var code = codeMirror.getValue();
    
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

function setSnippetCard(card, resp) {
  var response = JSON.parse(resp)
  var security = response['label'];
  var proba = response['proba'].toFixed(2);
  if (security == 1) {
    $(card).removeClass("card-default card-success");
    $(card).addClass("card-danger");
    $(card).find('p.title').text('Insecure Java Code: ' + proba)
  } else if (security == 0) {
    $(card).removeClass("card-default card-danger");
    $(card).addClass("card-success");
    $(card).find('p.title').text('Secure Java Code: '  + proba)
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
