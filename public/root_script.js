$(document).ready(function(){
    $('#retrieval_button').on('click',function(){
        console.log('func');
        var formdata = new FormData($(this).get(0));
        console.log(formdata);
        $.ajax({
            url  : "http://0.0.0.0:3001/predict",
            type : "POST",
            data : formdata,
            cache       : false,
            contentType : false,
            processData : false,
            // dataType    : "html"
        })
        .done(function(data, textStatus, jqXHR){
            alert(data);
        })
        .fail(function(jqXHR, textStatus, errorThrown){
            alert("fail");
        });
    });
});