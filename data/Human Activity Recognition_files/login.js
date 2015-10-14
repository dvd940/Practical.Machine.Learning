/**
 * 
 */
function submitViaEnter(evt) {
	evt = (evt) ? evt : event;
	var charCode = (evt.charCode) ? evt.charCode : ((evt.which) ? evt.which : evt.keyCode);
	if (charCode == 13) {
		try {
			$("btn_submit").click();
		} catch (e) {
			alert('erro ' + e);
		}
		return true;
	}
	return true;
}

function validarLogin(){
	$.ajax({
		url : "/doLogin",
		data : {p1: $("#url_origem").val(), loginname: $("#loginname").val(), loginpasswd: $("#loginpasswd").val()},
		type : 'POST',
		success : function(data) {
			if (data === "OK") {
				window.location = url;
			} else {
				alert(data);
			}
		}
	}); 
}

function logout(){
	$.ajax({
		url : "/doLogout",
		data : {p1: ""},
		type : 'POST',
		success : function(data) {
			window.location="/";
		}
	}); 
}

function gerarNovaSenha(){
	$.ajax({
		url : "/newPassword",
		data : {loginname: $("#loginname").val()},
		type : 'POST',
		success : function(data) {
				alert(data);
		}
	}); 
}