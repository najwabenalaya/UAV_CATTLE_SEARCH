one_static_search() {
  for s in 1 2 3 4 5 6 7 8 9 ; do
    #python stationary_target.py  --x-size 3 --y-size 2  --no-show --seed $s
    python stationary_target.py  --x-size 3 --y-size 3  --no-show --seed $s
    python stationary_target.py  --x-size 3 --y-size 4  --no-show --seed $s


  done
}


one_moving_search() {
  for s in 1 2 3 4 5 6 7 8 9 ; do
   python moving_target.py  --x-size 5 --y-size 6 --mobility 0.1  --max-time 30 --min-pd 0.9 --nb-poc 15 --no-show --seed $s
   python moving_target.py  --x-size 5 --y-size 4 --mobility 0.1  --max-time 20 --min-pd 0.9 --nb-poc 10 --no-show --seed $s
   python moving_target.py  --x-size 5 --y-size 2 --mobility 0.1  --max-time 10 --min-pd 0.9 --nb-poc 7 --no-show --seed $s

  done
}
static_search() {
  for s in 1 2 3 4 5 6 7 8 9 ; do
   tsp python stationary_target.py  --x-size 5 --y-size 6 --hover-time 1 --speed 1  --nb-poc 15 --no-show --seed $s
   tsp python stationary_target  --x-size 5 --y-size 4 --hover-time 1 --speed 1 --nb-poc 10 --no-show --seed $s
   tsp python stationary_target  --x-size 5 --y-size 2 --hover-time 1 --speed 1 --nb-poc 7 --no-show --seed $s


  done
}
two_search() {
  for s in 1 2 3 4 5 6 7 8 9 ; do
    python two_moving_target.py  --x-size 5 --y-size 6 --mobility-c1 0.1 --mobility-c2 0.1 --max-time 30 --min-pd 0.9 --nb-poc-c1 8 --nb-poc-c2 7 --no-show --seed $s
    python two_moving_target.py  --x-size 5 --y-size 4 --mobility-c1 0.1  --mobility-c2 0.1 --max-time 20 --min-pd 0.9 --nb-poc-c1 5 --nb-poc-c2 5 --no-show --seed $s
    #python two_moving_target.py  --x-size 5 --y-size 2 --mobility-c1 0.1  --mobility-c2 0.1 --max-time 7 --min-pd 0.9 --no-show --seed $s


  done
}


one_moving_search() {
  for s in 1 2 3 4 5 6 7 8 9 ; do
   python moving_target.py  --x-size 5 --y-size 6 --mobility 0.1  --max-time 30 --min-pd 0.9 --nb-poc 15 --no-show --seed $s
   python moving_target.py  --x-size 5 --y-size 4 --mobility 0.1  --max-time 20 --min-pd 0.9 --nb-poc 10 --no-show --seed $s
   python moving_target.py  --x-size 5 --y-size 2 --mobility 0.1  --max-time 10 --min-pd 0.9 --nb-poc 7 --no-show --seed $s

  done
}



two_search