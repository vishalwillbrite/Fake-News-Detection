const news = document.getElementById("news");
const count = document.getElementById("count");

if(news){

    news.addEventListener("input", function(){

        count.innerHTML = this.value.length + " Characters";

    });

}

function clearText(){

    news.value = "";

    count.innerHTML = "0 Characters";

}