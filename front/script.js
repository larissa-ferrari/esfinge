const sendRequest = () => {
    document.getElementById("responseContainer").innerHTML = "";
    // Captura o texto do textarea
    var texto = document.getElementById("userInput").value;

    // Dados para enviar na solicitação POST
    var data = {
        text: texto,
    };

    // Configuração da solicitação
    var requestOptions = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    };

    // Envia a solicitação
    fetch("http://127.0.0.1:5000/restore", requestOptions)
        .then((response) => response.json())
        .then((result) => {
            document.getElementById(
                "responseContainer"
            ).innerHTML = `<p>Texto Restaurado: ${result.restored_text}</p>
             <p>Região: ${result.region}</p>`;

            console.log(result);
        })
        .catch((error) => console.log("Erro:", error));
};
