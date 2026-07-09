const textarea = document.getElementById("news");
const charCount = document.getElementById("count");
const wordCount = document.getElementById("words");
const form = document.getElementById("newsForm");
const loading = document.getElementById("loading");

if (textarea) {

    textarea.addEventListener("input", () => {

        charCount.innerHTML = `${textarea.value.length} Characters`;

        const words = textarea.value
            .trim()
            .split(/\s+/)
            .filter(word => word.length > 0);

        wordCount.innerHTML = `${words.length} Words`;

    });

}

function clearText(){

    textarea.value="";

    textarea.dispatchEvent(new Event("input"));

}

function sampleNews(){

    textarea.value =
`NASA successfully launched a new satellite into orbit to improve global weather forecasting.`;

    textarea.dispatchEvent(new Event("input"));

}

if(form){

    form.addEventListener("submit", ()=>{

        loading.style.display="block";

    });

}