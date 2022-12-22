function toggleField(checkbox) {
    var field = document.getElementById(`${checkbox.value}-field`);
    if (checkbox.checked) {
      field.style.display = 'block';
    } else {
      field.style.display = 'none';
    }
  }
  function toggleInnerField(checkbox) {
    var field = document.getElementById(`${checkbox.value}-field`);
    if (checkbox.checked) {
      field.style.display = 'block';
    } else {
      field.style.display = 'none';
    }
  }
  function addFormRow() {
    // Get the form row container element
    const formRowContainer = document.getElementById('formRowContainer');
  
    // Get the form row template element
    const formRowTemplate = document.getElementById('formRowTemplate');
  
    // Create a copy of the form row template element
    const newFormRow = formRowTemplate.cloneNode(true);
  
    // Increment the ID of the select elements in the copy
    newFormRow.querySelector('#dataTypeSelect1').setAttribute('id', 'dataTypeSelect1' + (formRowContainer.children.length + 1));
    newFormRow.querySelector('#dataTypeSelect2').setAttribute('id', 'dataTypeSelect2' + (formRowContainer.children.length + 1));
  
    // Append the copy to the form row container element
    formRowContainer.appendChild(newFormRow);
  }
  function displaySelectedOptions(targetElementId) {
    // Get the select element
    const selectElement = document.getElementById('selectElement');
  
    // Get the selected options
    const selectedOptions = Array.from(selectElement.selectedOptions);
  
    // Get the target element
    const targetElement = document.getElementById(targetElementId);
  
    // Clear the target element
    targetElement.innerHTML = '';
  
    // Iterate over the selected options
    for (const option of selectedOptions) {
      // Check if the option is already displayed
      if (!targetElement.querySelector(`[data-value="${option.value}"]`)) {
        // Create a new element to display the option
        const optionElement = document.createElement('div');
        optionElement.textContent = option.textContent;
        optionElement.setAttribute('data-value', option.value);
  
        // Create a remove button for the option
        const removeButton = document.createElement('button');
        removeButton.innerHTML = '&times;';
        removeButton.classList.add('btn', 'btn-secondary', 'btn-sm');
        removeButton.setAttribute('onclick', `removeDisplayedOption('${option.value}', '${targetElementId}')`);
  
        // Append the remove button to the option element
        optionElement.appendChild(removeButton);
  
        // Append the option element to the target element
        targetElement.appendChild(optionElement);
      }
    }
  
    // Display the selected options as an array
    targetElement.innerHTML += `[${selectedOptions.map(option => option.textContent).join(', ')}]`;
  }
  
  function removeDisplayedOption(optionValue, targetElementId) {
    // Get the target element
    const targetElement = document.getElementById(targetElementId);
  
    // Find the option element to remove
    const optionElement = targetElement.querySelector(`[data-value="${optionValue}"]`);
  
    // Remove the option element
    targetElement.removeChild(optionElement);
  }
  