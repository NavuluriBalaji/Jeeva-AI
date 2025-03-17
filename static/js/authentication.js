document.getElementById('hospitalManagementBtn').addEventListener('click', (event) => {
    event.preventDefault(); // Prevent the default link behavior
    document.getElementById('authModal').style.display = 'block'; // Show the authentication modal
});

document.getElementById('authForm').addEventListener('submit', async (event) => {
    event.preventDefault(); // Prevent the default form submission

    const userName = document.getElementById('userName').value;
    const userId = document.getElementById('userId').value;

    try {
        const response = await fetch('/authenticate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ userName, userId })
        });

        const result = await response.json();

        if (result.success) {
            window.location.href = 'http://127.0.0.1:8000/';
        } else {
            alert('User is unauthorized');
        }
    } catch (error) {
        console.error('Error during authentication:', error);
        alert('An error occurred while trying to authenticate.');
    }
});
