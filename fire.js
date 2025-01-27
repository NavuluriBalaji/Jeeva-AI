import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.8.0/firebase-app.js'
import { getAuth } from 'https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js'
import { getDatabase, ref as dbRef, push as dbPush, onValue } from 'https://www.gstatic.com/firebasejs/10.8.0/firebase-database.js'
import { getStorage } from 'https://www.gstatic.com/firebasejs/10.8.0/firebase-storage.js'

const firebaseConfig = {
    apiKey: "yourapikey",
    authDomain: "authentication-d3540.firebaseapp.com",
    databaseURL: "https://authentication-d3540-default-rtdb.firebaseio.com",
    projectId: "authentication-d3540",
    storageBucket: "authentication-d3540.appspot.com",
    messagingSenderId: "241233830169",
    appId: "1:241233830169:web:954daa7b875f5108722588"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getDatabase(app);
const storage = getStorage(app);

export { auth, app, db, dbRef, dbPush, onValue, storage }
