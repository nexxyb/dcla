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
  
  function updateTaskProgress(options) {
    // send an AJAX request to the Django view function to check the task progress
    $.ajax({
        url: '/task-progress/' + options.taskIdPrefix,
        type: 'GET',
        dataType: 'json',
        success: function(response) {
            var status = response.status;
            var progress = response.progress;
            // update the progress bar based on the task status and progress
            if (status == 'SUCCESS') {
                // task is complete, update the progress bar to 100%
                document.getElementById(options.progressBarId).style.width = '100%';
                document.getElementById(options.progressBarId + '-status').innerHTML = 'Task Completed';
            } else if (status == 'FAILURE') {
                // task has failed, stop checking the status and display an error message
                clearInterval(options.intervalId);
                document.getElementById(options.progressBarId + '-status').innerHTML = 'Task failed';
            } else {
                // task is still running, update the progress bar based on the task's progress
                document.getElementById(options.progressBarId).style.width = progress + '%';
                document.getElementById(options.progressBarId + '-status').innerHTML = 'Task in Progress';
            }
        }
    });
}
 
function startTaskPolling(element) {
    // get the progress bar ID from the element's ID
    var progressBarId = element.id.replace('-container', '');

    // check if the task ID is present in the session
    var taskIdPrefix = progressBarId + '_id';
    //  var taskId = '{{ request.session.[' + taskIdPrefix + '] }}';
    // if (!taskId) {
    //     // task ID is not present, hide the progress bar and return
    //     element.style.display = 'none';
    //     return;
    // }

    // start polling the task status every 1000 milliseconds (1 second)
    var intervalId = setInterval(function() {
        updateTaskProgress({
            progressBarId: progressBarId,
            taskIdPrefix: taskIdPrefix,
            intervalId: intervalId,
        });
    }, 1000);
}
