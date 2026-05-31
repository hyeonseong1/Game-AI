var now, last;
var requestId;
var keySettings = () => {};
var activePlayers = [];

function postToParent(payload) {
    if (window.parent && window.parent !== window) {
        window.parent.postMessage({ source: "jstetris", ...payload }, "*");
    }
}

const init = () =>
{
    resize();

    loadSettings();

    document.querySelector("#keybinding").addEventListener("click", event=>{
        if(event.target && event.target.nodeName == "TD")
        {
            if(event.target.id)
            {
                clearClicked();
                event.target.className += " clicked"

                document.onkeydown = (e) =>
                {
                    if(e.keyCode == 27)
                    {
                        clearClicked();
                        document.onkeydown = null;
                        return;
                    }
                    let temp;
                    switch(e.keyCode)
                    {
                        case 17: temp = "CTRL";     break;
                        case 21: temp = "R ALT";    break;
                        case 25: temp = "HANJA";    break;
                        case 32: temp = "SPACE";    break;
                        case 37: temp = `←`;       break;
                        case 38: temp = `↑`;       break;
                        case 39: temp = `→`;       break;
                        case 40: temp = `↓`;       break;
                        default: temp = e.key;
                    }
                    event.target.innerText = temp;
                    keySettings[event.target.id] = e.keyCode;
                    if(storageAvailable()) localStorage[event.target.id] = e.keyCode;
                }
            }   
        }
    });  

    const root = document.getElementById("jstetris-root");
    if (root && typeof ResizeObserver !== "undefined") {
        new ResizeObserver(resize).observe(root);
    }
    window.addEventListener("resize", resize, false);
}

const gameStart = () => {
    const menu = document.getElementById("main");
    if (menu) menu.hidden = true;

    if (requestId) {
        cancelAnimationFrame(requestId);
        requestId = undefined;
    }

    changeKeyBindings();

    let players = [];
    let playerNum = settings[1] + 1;
    let randomEngine = new random();
    for (let i = 0; i < playerNum; i++) {
        players.push(new player(i, randomEngine));
    }
    activePlayers = players;
    for (let i = 0; i < playerNum; i++) {
        players[i].countDown();
    }
    setTimeout(() => {
        now = last = timeStamp();
        requestId = requestAnimationFrame(() => {
            animate(players);
        });
    }, 3000);
};
/**
 * Calculates delta time between requestAnimationFrame calls and
 * passes the resulting dt (normalized to 1s) to the update function.
 */
const animate = (players) => {
    let playerNum = settings[1] + 1;

    now = timeStamp();
    var dt = (now - last) / 1000.0;
    last = now;

    for (var i = 0; i < playerNum; i++) {
        players[i].update(Math.min(1, dt));
        if (players[i].gameOver) {
            cancelAnimationFrame(requestId);
            requestId = undefined;
            postToParent({
                type: "gameover",
                score: players[i].stg.score,
                level: players[i].stg.getLevel(),
                player: i,
            });
            if (window.parent === window) {
                alert(DEATH_MESSAGE(i));
            }
            return;
        }
    }

    if (players[0]) {
        postToParent({
            type: "score",
            score: players[0].stg.score,
            level: players[0].stg.getLevel(),
        });
    }

    if (requestId !== undefined) {
        requestId = requestAnimationFrame(() => {
            animate(players);
        });
    }
};

const toggleSettings = () =>
{
    var a = document.getElementById("settings");
    a.hidden = !a.hidden
}

const settingsButton = (index, lr) =>
{
    settings[index] += (lr==1)?1:-1;
    let a = 0;
    switch(index)
    {
        case 0:
            a = 3;
            break;
        case 1:
            a = 2;
            break;
    }

    settings[index] = Math.abs(settings[index])%a;
    
    switch(index)
    {
        case 0:
            document.getElementById("GAMEMODE_GOALS").innerText = GAMEMODE_NAMES[settings[index]];
            break;
        case 1:
            document.getElementById("GAMEMODE_PLAYER").innerText = settings[1]+1;
            break;
    }

    localStorage['gameSettings'] = settings;
}

const timeStamp = () =>
{
    return new Date().getTime();
}

const resize = () => {
    const root = document.getElementById("jstetris-root");
    const ratio = canvas.width / canvas.height;
    let cw = root ? root.clientWidth : window.innerWidth;
    let ch = root ? root.clientHeight : window.innerHeight;
    if (!cw || !ch) return;

    let w = cw;
    let h = Math.floor(w / ratio);
    if (h > ch) {
        h = ch;
        w = Math.floor(h * ratio);
    }
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    canvas2.style.width = w + "px";
    canvas2.style.height = h + "px";
    canvas3.style.width = w + "px";
    canvas3.style.height = h + "px";
};

const clearClicked = () => 
{
    let nodes = document.querySelector("#keybinding").querySelectorAll(".keybinding.clicked");
    for(let node of nodes)
    {
        node.className = "keybinding";
    }
}

const changeKeyBindings = () =>
{
    for(const property in keySettings)
    {
        var temp = property.split('_')
        KEY[temp[0]][temp[1]] = keySettings[property]
        if(storageAvailable())
        {
            localStorage[property] = keySettings[property];
        }
    }
    
    MOVES[KEY.p1.LEFT] =  p=>({...p, x: p.x-1, lastMove: LAST_MOVE.MOVE}),
    MOVES[KEY.p1.RIGHT] = p=>({...p, x: p.x+1, lastMove: LAST_MOVE.MOVE}),
    MOVES[KEY.p1.DOWN] =  p=>({...p, y: p.y+1, lastMove: LAST_MOVE.DOWN}),
    MOVES[KEY.p2.LEFT] = p=>({...p, x: p.x-1, lastMove: LAST_MOVE.MOVE}),
    MOVES[KEY.p2.RIGHT] = p=>({...p, x: p.x+1, lastMove: LAST_MOVE.MOVE}),
    MOVES[KEY.p2.DOWN] =  p=>({...p, y: p.y+1, lastMove: LAST_MOVE.DOWN})
}

const loadSettings = () =>
{
    if(storageAvailable())
    {
        for (const property in localStorage)
        {
            const target = document.querySelector('#'+property);
            let temp;
            key = parseInt(localStorage[property]);
            if(target !== null) 
            {
                switch(key)
                {
                    case 17: temp = "CTRL";     break;
                    case 21: temp = "R ALT";    break;
                    case 25: temp = "HANJA";    break;
                    case 32: temp = "SPACE";    break;
                    case 37: temp = `←`;       break;
                    case 38: temp = `↑`;       break;
                    case 39: temp = `→`;       break;
                    case 40: temp = `↓`;       break;
                    default: temp = String.fromCharCode(localStorage[property]);
                }
                target.innerText = temp;
                keySettings[property] = localStorage[property];    
            }
        } 
        const savedSettings = localStorage['gameSettings']
        if(savedSettings)
        {
            let temp = savedSettings.split(',');
            for(let i = 0; i<temp.length;i++)
                settings[i] = parseInt(temp[i])
                
            document.getElementById("GAMEMODE_GOALS").innerText = GAMEMODE_NAMES[settings[0]];
            document.getElementById("GAMEMODE_PLAYER").innerText = settings[1]+1;
        }
    }
}

const storageAvailable = () =>{
    try{
        return localStorage !== null;
    } catch (e) {
        console.error(e);
        return false;
    }
}