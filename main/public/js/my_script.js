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

$(function() {
    $('input[id=upload]').change(function() {// upするinputのID
        var file = $(this).prop('files')[0];
        if (! file.type.match('image.*')) {
            $(this).val('');
            $('#img1').html('');
            return;
        }
        var reader = new FileReader();
        reader.onload = function() {
            $('#img1').attr('src',reader.result);
        }
        reader.readAsDataURL(file);
    });
});