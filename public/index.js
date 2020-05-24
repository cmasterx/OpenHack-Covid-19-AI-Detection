function openAboutUsModal()
{

    let checkboxTicked = localStorage.getItem("aboutUsModal") === "false" ? true : false;
    $('#neverShowCheckbox').prop('checked', checkboxTicked);

    $('#aboutUsModal').modal('show');
}

function closeAboutUsModal()
{
    $('#aboutUsModal').modal("hide");

    // let ticked = $('#neverShowCheckbox').is(':checked') ? "false" : "true";
    // localStorage.setItem("introModal", ticked);
}
