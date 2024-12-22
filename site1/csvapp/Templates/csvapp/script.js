// JavaScript logic with validation and effects
document.addEventListener('DOMContentLoaded', function () {
    // Fade-in effect
    const formContainer = document.querySelector('.form-container');
    formContainer.classList.add('visible');  // Trigger fade-in effect

    // Form submission event
    document.getElementById('loginForm').addEventListener('submit', function (e) {
        e.preventDefault(); // Prevent the form from submitting

        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;
        const errorMessage = document.getElementById('errorMessage');
        const loader = document.getElementById('loader');
        const btnText = document.getElementById('btnText');

        // Clear error message initially
        errorMessage.style.display = "none";

        // Simple username validation
        if (!validateInput(username)) {
            displayMessage("Vui lòng nhập tên đăng nhập.", true);
            return;
        }

        // Check for strong password criteria
        if (!validatePassword(password)) {
            displayMessage("Mật khẩu phải có ít nhất 8 ký tự, bao gồm chữ cái hoa, số và ký tự đặc biệt.", true);
            return;
        }

        // Show loader and simulate login validation
        loader.style.display = "block";
        btnText.style.display = "none";

        setTimeout(() => {
            loader.style.display = "none";
            btnText.style.display = "block";

            if (username === "admin" && password === "passworD@123") {
                alert("Đăng nhập thành công!");
                // Redirect to another page or take appropriate action
                window.location.href = "index.html";
            } else {
                displayMessage("Tên đăng nhập hoặc mật khẩu không hợp lệ.", true);
            }
        }, 2000);
    });

    // Toggle Dark Mode
    document.querySelector('.toggle-btn').addEventListener('click', toggleDarkMode);
});

// Toggle Dark Mode
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
}

// Toggle password visibility
document.getElementById('togglePassword').addEventListener('click', function () {
    const passwordInput = document.getElementById('password');
    const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
    passwordInput.setAttribute('type', type);
    this.classList.toggle('fa-eye-slash');
});

// Function to display messages with a timeout
function displayMessage(message, isError = false) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorMessage.style.display = "block";
    errorMessage.style.transition = "opacity 0.5s";
    if (isError) {
        errorMessage.style.color = "red";
    } else {
        errorMessage.style.color = "green";
    }

    // Automatically hide the message after 3 seconds
    setTimeout(() => {
        errorMessage.style.display = "none";
    }, 3000);
}

// Function to handle input focus
document.querySelectorAll('input').forEach(input => {
    input.addEventListener('focus', function () {
        this.style.borderColor = '#0056b3';
        this.style.boxShadow = '0 0 5px rgba(0, 123, 255, 0.5)';
    });
    input.addEventListener('blur', function () {
        this.style.borderColor = '#007bff';
        this.style.boxShadow = 'none';
    });
});

// Input validation functions
function validateInput(input) {
    return input.trim() !== "";
}

function validatePassword(password) {
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);
    return password.length >= 8 && hasUpperCase && hasLowerCase && hasNumbers && hasSpecialChar;
}

// Add event listeners for inputs
document.querySelectorAll('input').forEach(input => {
    input.addEventListener('input', function () {
        if (!validateInput(this.value)) {
            displayMessage("Trường này không được để trống", true);
        } else {
            document.getElementById('errorMessage').style.display = "none"; // Clear error if valid
        }
    });
});
