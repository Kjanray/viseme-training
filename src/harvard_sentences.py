"""
Harvard/IEEE Sentence Lists
=============================
720 phonetically balanced sentences from IEEE Std 1969 (public domain).
72 lists x 10 sentences each, designed for speech intelligibility testing.

Each list covers all English phonemes in balanced proportions, making them
ideal for training viseme-to-blendshape mappings with full coverage.

Split strategy (by list number, preserving phoneme distribution):
  Train: Lists 1-57  (570 sentences, 79%)
  Val:   Lists 58-64 (70 sentences, 10%)
  Test:  Lists 65-72 (80 sentences, 11%)
"""

from collections import Counter

# fmt: off
# ===================================================================
# All 72 lists (720 sentences)
# Source: IEEE Recommended Practice for Speech Quality Measurements
# ===================================================================

HARVARD_SENTENCES = [
    # List 1
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    # List 2
    "The beauty of the view stunned the young boy.",
    "The source of the huge river is the clear spring.",
    "The salt breeze came across from the sea.",
    "The current drove the small boat upstream.",
    "Drop the ashes on the worn old rug.",
    "The child crawled into the dense grass.",
    "Boards served in place of chairs and tables.",
    "The paste in the can hardened over night.",
    "A rag will soak up spilled water.",
    "Send the stuff in a large paper bag.",
    # List 3
    "The clock struck to mark the third period.",
    "A sealed jar of rum was in the cargo hold.",
    "It's a dense crowd in the big stores.",
    "His hip struck the edge of the counter.",
    "Lift the square stone over the fence.",
    "The rope will bind the seven books at once.",
    "Hop over the fence and plunge in.",
    "The friendly gang left the drug store.",
    "Mesh mends the gap in the old cloth.",
    "The loss of the second hand made the watch useless.",
    # List 4
    "The bark peeled off the old oak tree.",
    "Cats and dogs each hate the other.",
    "The pipe began to rust while new.",
    "Open the crate but don't break the glass.",
    "Add the sum to the product of these three.",
    "Thieves who rob friends deserve jail.",
    "The ripe taste of cheese improves with age.",
    "Act on the plan that seems best to you.",
    "The hostess met her guests at the door.",
    "The blouse had a stain but was not ruined.",
    # List 5
    "A mud house is built on stiff pine poles.",
    "The best play of the season has just started.",
    "A round shape is preferred by most builders.",
    "The first part of the plan needs changing.",
    "The music roared through the crowded room.",
    "The brown sack hung to the side of the machine.",
    "The youth drove with zest but little skill.",
    "The key you made won't fit this lock.",
    "Breathe deep and smell the pine air.",
    "A flat tire and no spare made the car useless.",
    # List 6
    "A gold vase is both rare and costly.",
    "The long journey home took a year.",
    "She floated on the wide calm lake.",
    "A strong bid knocked the price up.",
    "Jerk the rope and the old bell rings.",
    "A king ruled the state in the early days.",
    "The ship was torn apart on the sharp reef.",
    "Soak the cloth and wring it out.",
    "The broom swept the dirt from the floor.",
    "The small red book was her delight.",
    # List 7
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The colt reared and threw the tall rider.",
    "It snowed, rained, and hailed the same morning.",
    "Read verse out loud for pleasure.",
    "Hoist the load to your left shoulder.",
    "Take the winding path to reach the lake.",
    "Note closely the size of the gas bill.",
    "Wipe the grease off his dirty face.",
    "Mend the coat before you go out.",
    # List 8
    "The wrist was badly strained and hung limp.",
    "The stray cat gave birth to kittens.",
    "The young girl gave no clear response.",
    "The meal was cooked before the bell rang.",
    "What joy there is in living.",
    "A king once sat on a golden throne.",
    "The press felt the charm of the new town.",
    "The petty cash was nearly spent.",
    "The look in his eye told the whole story.",
    "The new drug proved to be quite effective.",
    # List 9
    "Mud was spattered on the front of his white shirt.",
    "The blind man counted his old coins.",
    "A siege will crack the strong defense.",
    "Grape juice and water mix well together.",
    "Roads are paved with sticky tar.",
    "Fake gems don't shine in the dark.",
    "The drip of the rain made a pleasant sound.",
    "Smoke poured out of every crack.",
    "The crooked path wound through the thick woods.",
    "The rug was cleaned on both sides.",
    # List 10
    "A cramp is no small danger on a swim.",
    "He said a curt word to the sly old man.",
    "The stale smell of old beer lingers.",
    "The desk was firm on the shaky floor.",
    "It takes heat to bring out the odor.",
    "A child could spin these small toy tops.",
    "Slide the tray across the glass top.",
    "The cloud moved in a stately way and was gone.",
    "Light maple sugar is hard to come by.",
    "We go to the movies every night.",
    # List 11
    "The dime was lost in the thick mud.",
    "The sack of potatoes had to be moved.",
    "The bank pressed for payment of the debt.",
    "The child pet the big furry cat.",
    "Her entry was a joy to all of them.",
    "The perch leaped from the bank into the cool lake.",
    "There is a strong bond between us.",
    "The team of mules pulled the heavy load.",
    "The sun dropped behind a thick cloud.",
    "An evening stroll along the canal is pleasant.",
    # List 12
    "The bill was paid every third week.",
    "Cut the cord to release the big bundle.",
    "His chum chose to make a costly purchase.",
    "The fur of cats goes by many names.",
    "Back the car into the slot near the curb.",
    "Rain swept along the old highway.",
    "Clams are small, round, soft, and tasty.",
    "The fan whirred its round blades softly.",
    "The line joined the two large rivers.",
    "The prince ordered his head cut off.",
    # List 13
    "The red tape bound the smuggled food.",
    "Time brings all things to a close.",
    "The marble pillars were split in two.",
    "Try to have the parts ready on time.",
    "Peel the paint off every old board.",
    "The quick fox jumped on the sleeping cat.",
    "The nozzle of the fire hose was bright and clean.",
    "Scorching heat can be hard to bear.",
    "The young prince became heir to the throne.",
    "He sent the figs but kept the ripe cherries.",
    # List 14
    "The roof should be tilted at a sharp slant.",
    "The old song came to mind during the concert.",
    "That move means the game is over.",
    "Fill the ink jar with sticky glue.",
    "He smoke a big pipe with strong contents.",
    "Screen the porch with a wire net.",
    "Books can be a hard road to knowledge.",
    "The pods clung tight to the bush stems.",
    "Shut the hatch before the waves push it in.",
    "The odor of spring makes young hearts jump.",
    # List 15
    "The curved road wound through the mountain pass.",
    "The child ran along the dusty road.",
    "The corn has turned golden brown.",
    "A whistle sounded and the game started.",
    "The wagon moved on well oiled wheels.",
    "March the soldiers past the front gate.",
    "A cup of sugar makes sweet fudge.",
    "Place a rosebud near the front window.",
    "Two pins were stuck in the dark cushion.",
    "His shirt was clean but wrinkled badly.",
    # List 16
    "A thick coat of black paint covered all.",
    "The child hit the dog on its paw.",
    "Cars and buses stalled in snow drifts.",
    "The set of china hit the floor with a crash.",
    "This is a grand season for hikes on the road.",
    "The dune rose from the edge of the water.",
    "Those words were the cue for the actor to leave.",
    "A yacht slid around the point into the bay.",
    "The two old men quarreled often.",
    "The tube was blown and the tire flat.",
    # List 17
    "A wisp of hair fell over his muddy forehead.",
    "Bail the boat to stop it from sinking.",
    "The key to the cabinet hung on a brass ring.",
    "The child cried and kicked the top of his bed.",
    "Push the door open and walk into the room.",
    "Flood the mails with requests for bounty.",
    "The pencil was cut to be sharp at both ends.",
    "The gate swung open on the rusty hinges.",
    "Dimes showered down from all sides.",
    "Pick a card and slip it under the pack.",
    # List 18
    "A round mat fits neatly in the corner.",
    "The old man sat near the front of the church.",
    "A bean stalk grew into a tall tree.",
    "The child sat in a corner and whimpered.",
    "We took our text from the gospel of Mark.",
    "The hedge wound around the side of the hill.",
    "A panel of cloth was a prized find.",
    "The vast empty desert glistened in the sun.",
    "Each penny shone like new.",
    "The rude remark drew a sharp response.",
    # List 19
    "The horn of the car woke the sleeping cop.",
    "The heart beat strongly and with firm strokes.",
    "The pearl was worn in a thin silver ring.",
    "The fruit of a fig tree is quite sweet.",
    "Corn stalks stand in bunches near the fence.",
    "The steep trail is hard for our crew to climb.",
    "The store was jammed before the sale could start.",
    "It was a bad error on the part of the new judge.",
    "Strong text goes on a shelf at the top.",
    "The nets trapped the schools of trout.",
    # List 20
    "Two plus seven is less than ten.",
    "The cows grazed on the open meadow.",
    "Both brothers wear the same size.",
    "In some form or other we all are beggars.",
    "The swift stream raced past us.",
    "Hedge the road to stop the drift of sand.",
    "She sewed the bright dress on a dark background.",
    "We dressed the tree on Christmas Eve.",
    "The best method is to fix it in place with clips.",
    "If you mumble your speech will be lost.",
    # List 21
    "At that high level the air is pure.",
    "Drop the two when you add the figures.",
    "A file was used to smooth the rough edges.",
    "The cold drizzle will halt the bond drives.",
    "Grace makes the girl that much prettier.",
    "Fake text is of no use to a scholar.",
    "Seven seals were stamped on the bright badge.",
    "Our troops are set to strike at dawn.",
    "A theft was made in a well guarded room.",
    "The plan was formed to create a new group.",
    # List 22
    "The beam dropped down on the worker's head.",
    "A thing of beauty is truly a joy forever.",
    "North winds bring colds and fevers.",
    "He slowly drank the strong dark coffee.",
    "Move the vat over the hot fire.",
    "The rod was used to prop the tent flap up.",
    "He lay prone and hardly moved a limb.",
    "Tin cans are handy to have around the house.",
    "The firm owns many acres of good farmland.",
    "A six comes up more often than a ten.",
    # List 23
    "Fly by night if you can stand the cold.",
    "Thick text needs special decoding.",
    "He found a tax form in the tax office drawer.",
    "Empty pockets don't ever make the grade.",
    "She was thrifty and spent little on new things.",
    "A couch can be made from a thick dark blanket.",
    "The hapless players were trapped and soaked.",
    "The vast waste land stretched for miles.",
    "The club members joined to clean their park.",
    "The leaf drifted along with a slow spin.",
    # List 24
    "A white silk jacket goes with any shoes.",
    "A break in the dam almost flooded the town.",
    "Greet the new guests and make them welcome.",
    "When the frost comes thick the water stands still.",
    "A sash window covers half the door.",
    "The child crept closer and hugged the dog.",
    "The red ball rolled past his feet.",
    "A toad and a frog are hard to tell apart.",
    "Pick the best card and put the rest down.",
    "Her purse had the money for the fare.",
    # List 25
    "The small red neon light went out.",
    "The old pan was covered with hard bite marks.",
    "The broad road shimmered in the hot sun.",
    "The frosty air passed through every crack.",
    "He offered proof in the form of a large chart.",
    "Send the book to the other end of the store.",
    "The dusty bench stood by the stone wall.",
    "The square block slid over the smooth surface.",
    "The bird flew over the high fence.",
    "Next Sunday is the twelfth of the month.",
    # List 26
    "The three story house was built of stone.",
    "The white plaster made a poor background.",
    "He dressed in an old coat and dirty pants.",
    "The small boat turned over in the heavy swell.",
    "She was dressed in a purple gown.",
    "Sip the broth and prepare for bed.",
    "The man leaped from the door of the plane.",
    "A shower of rain fell from the dark sky.",
    "He crept through the narrow lane without care.",
    "The grass curled around the base of the fence post.",
    # List 27
    "The logs fell and tumbled into the clear stream.",
    "The girl at the booth sold fifty bonds.",
    "The deep gorge was bare and dark.",
    "A roll of cloth was lying by the wall.",
    "The poor child slept on the hard ground.",
    "Pack the records in a neat thin case.",
    "The gold ring fits only a thin finger.",
    "The long pipe ran almost the length of the ditch.",
    "It was hidden from sight by a wild tangle of vines.",
    "The wall phone rang loud and often.",
    # List 28
    "We find joy in the simplest things.",
    "Type the paper with care and speed.",
    "The child drew pictures on the ground.",
    "The wreck of the ship sank to the floor of the sea.",
    "There is more than one way to cook a duck.",
    "The point of the steel pen was bent and dull.",
    "The straw nest housed five new kittens.",
    "The quick run made him short of breath.",
    "The fly sat on the lip of the old cup.",
    "Clothes and lodging are free to new members.",
    # List 29
    "We heard the sad news quite by chance.",
    "The lease ran out in sixteen weeks.",
    "A taut rope can often make a bridge.",
    "Fasten two pins on each side.",
    "A cold dip restores health and zest.",
    "He carved a head from a block of dark wood.",
    "She wore a plain blue dress to the party.",
    "The ramp led up to the wide door.",
    "It's not fair to trick a sleeping dog.",
    "The dress was torn along the front seam.",
    # List 30
    "The wretch is in a foul cell at best.",
    "A plate of stew can make a fine meal.",
    "The facts were clear and easily checked.",
    "We fished at the side of the pond.",
    "The colt broke short and threw the jockey.",
    "Plead to the court and the fine will drop.",
    "Calves thrive on tender spring grass.",
    "Post no bills on this tall fence.",
    "Dunk the stale biscuits into strong drink.",
    "A good book informs of what we ought to know.",
    # List 31
    "The small boy stole the food from the store.",
    "The fight will end in just six minutes more.",
    "The store walls were lined with rolls of cloth.",
    "The wagon stopped at the edge of the field.",
    "The old clothes were dirty and torn.",
    "The show was a hit from the very start.",
    "Bring your best pens to the meeting.",
    "His shirt was red and his coat blue.",
    "A thin book rests at the top of the shelf.",
    "Cats sleep well on a warm bed.",
    # List 32
    "The play seemed dull and quite stupid.",
    "The land was bare and parched by the sun.",
    "The girl danced jigs on the steep deck.",
    "Some men crave food while others crave love.",
    "He ran half way to the hardware store.",
    "The wide road shimmered in the hot sun.",
    "The cloud moved in a stately way and was gone.",
    "Sit on the perch and sunbathe daily.",
    "The ink stain dried on the finished page.",
    "The walled town was seized without a fight.",
    # List 33
    "The bugs flew into the bright porch light.",
    "The hay was cut and stacked in the barn.",
    "A strong arm and a quick eye are needed.",
    "Girls love to play with dolls and comb their hair.",
    "Some food tastes good with a trace of nutmeg.",
    "A pound of sugar costs more than eggs.",
    "The sky in the west is tinged with orange red.",
    "The bombs left most of the town in ruins.",
    "Pour the stew into a deep white dish.",
    "The sun comes up from behind the tall mountain.",
    # List 34
    "The doctor cured him with these pills.",
    "The new girl was fired at the end of the week.",
    "The third act was dull and tired the audience.",
    "A young child should not suffer fright.",
    "The marsh will freeze when cold enough.",
    "The rug was thick and red in color.",
    "He wrote his last novel there at the inn.",
    "The loss was one that he could not bear.",
    "The cow was found tied to a fence post.",
    "Bail the water from the leaky boat.",
    # List 35
    "His plan is to build a new house next year.",
    "A lathe shapes brass into a round form.",
    "The zinc bar helped shield the acid fumes.",
    "Slide the catch back and open the desk.",
    "Help the weak to preserve their strength.",
    "A sulky child is a bad playmate.",
    "The dusty road gave off clouds of fine sand.",
    "Pack the kits and don't forget the salt.",
    "The desk and both chairs were painted tan.",
    "Throw your trash in the garbage pail.",
    # List 36
    "The cleat snagged his coat on the sharp turn.",
    "Boards cut thin are handy to use.",
    "He left the gas range heater on.",
    "The cup cracked and spilled its contents.",
    "Paste can cleanse the most dirty brass.",
    "Sell your gift to a buyer at a good price.",
    "The cast played for a sellout audience.",
    "The four men rode the bus to the port.",
    "They told wild tales to frighten him.",
    "The lock on the jar was hard to open.",
    # List 37
    "The chap slipped into the crowd and was lost.",
    "The ground hardened when the cold wind swept across.",
    "The cement block wall did not crack.",
    "Pack the plush robes in the worn suitcase.",
    "His plan meant nothing to the wise fool.",
    "Brass rings are sold by these street vendors.",
    "The child cried and kicked the top of his bed.",
    "The reason for his daze was beyond all guess.",
    "Sixty four comes before sixty five.",
    "March along the dusty road at once.",
    # List 38
    "A yacht slid around the point into the bay.",
    "He smoked a pipe with strong contents.",
    "Thick text needs special decoding.",
    "The early morning mist hung low over the field.",
    "The new girl was fired at the end of the week.",
    "The lamp shone with a steady green flame.",
    "Take the match and strike it against your shoe.",
    "The pipe ran almost the length of the ditch.",
    "The punch bowl was full of melting ice.",
    "He offered proof in the form of a large chart.",
    # List 39
    "Add the column and jot down the sum.",
    "We found the road was a smooth ride.",
    "Say it slowly but make it ring clear.",
    "The stiff text was bound in dark blue leather.",
    "He picked up the dice for a second roll.",
    "These text lines should be read from the bottom.",
    "The sense of smell is better than that of touch.",
    "He was sly and evil but only on the outside.",
    "Down that road is the way to the grain farmer.",
    "The heap of things held the clue we sought.",
    # List 40
    "Dots of light betrayed the black submarine.",
    "Read every line from the top to the end.",
    "The cozy chair felt like old times.",
    "A yacht is a beautiful sight on a calm sea.",
    "The hose ran over the edge of the ditch.",
    "Those last words were a strong statement.",
    "He wrote down a long list of items.",
    "A glass tumbler holds more than one pint.",
    "Twist the valve and shut off the steam.",
    "The cane helped him walk to the next street.",
    # List 41
    "Fasten two pins on each side.",
    "He dodged the ball that was kicked at him.",
    "Slide the box into that empty space.",
    "The bark peeled off the old oak tree.",
    "Jangle the keys so that she may hear.",
    "A hedge between keeps friendship green.",
    "Cut the pie into large parts.",
    "Men strive but seldom get rich.",
    "Always close the barn door tight.",
    "He lay face down and cried in shame.",
    # List 42
    "Even the worst will beat his low score.",
    "Wake and rise and step into the green outdoor.",
    "The whiff of pine odor stung his nose.",
    "There is no trade for the man without talent.",
    "It is hard to erase blue or red ink.",
    "The cap fitted over the head just right.",
    "Pages bound in cloth make a good book.",
    "A line of ants crossed the wide mat.",
    "The burst of wind moved the tabletop.",
    "A round cheese was set on the shelf.",
    # List 43
    "A lame back kept his score low.",
    "The child ate his meal then ran outside.",
    "The tree was struck by a bolt of lightning.",
    "The tin box held cheap and useless articles.",
    "Stop whistling and hear the song of the birds.",
    "The chair looked strong but had no bottom.",
    "A steep trail is bad for loaded mules.",
    "The dry leaves were tossed about by the wind.",
    "The child wept when his toy broke.",
    "Pluck the bright rose without leaves.",
    # List 44
    "Bribes fail where good men work.",
    "We need grain to keep our mules healthy.",
    "Pack the records in a neat thin case.",
    "The choice of war or peace lies with us.",
    "A thin disk of glass broke in his hand.",
    "He sent the boy on a short errand.",
    "The sun shone and the rain fell in gentle drops.",
    "When it gets dark she lights the lamp.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    # List 45
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    "Hot water is preferred to cold for washing.",
    "Thick text needs more time to read.",
    "Tend to the work at hand.",
    "The youth drove with zest but little skill.",
    "Use a pencil to write the first draft.",
    # List 46
    "He ran half way to the hardware store.",
    "The clock struck to mark the third period.",
    "A small creek cut across the field.",
    "Cars and buses stalled in snow drifts.",
    "The set of china hit the floor with a crash.",
    "A bit of dust fell from the old rug.",
    "A dark cloud moved in from the south.",
    "The stiff crust of bread held a small seed.",
    "The wreck was found right after the storm.",
    "Dip the pail once and let it settle.",
    # List 47
    "A fizz from a leak hissed across the room.",
    "The tin plates were sharp and held no food.",
    "Sickness kept him home the third week.",
    "The wide road shimmered in the hot sun.",
    "The frosty air passed through every crack.",
    "He sat still and listened to the sounds.",
    "Drop the red book on top of the desk.",
    "Four hours of steady work faced us.",
    "The hat brim was wide and sheltered his face.",
    "The grass waved along the hillside.",
    # List 48
    "Keep the hatch tight and the watch constant.",
    "The priest made him give back the ring.",
    "Two blue fish swam in the tank.",
    "Her purse was full of useless trash.",
    "The stump of the tree was old and rotten.",
    "The dreary rain made the grey day dull.",
    "Leaves turn brown and yellow in the fall.",
    "The nest of the bird was found in the tree.",
    "A dash of pepper spoils beef stew.",
    "A zest for life keeps you young.",
    # List 49
    "A strong bid knocked the price up.",
    "The long journey home took a year.",
    "She floated on the wide calm lake.",
    "A gold vase is both rare and costly.",
    "Jerk the rope and the old bell rings.",
    "Mend the cloak before the cold wind comes.",
    "The small red book was her delight.",
    "Soak the cloth and wring it out.",
    "The broom swept the dirt from the floor.",
    "The ship was torn apart on the sharp reef.",
    # List 50
    "Crouch before you jump or miss the mark.",
    "Pack the box tight with the five dozen jugs.",
    "This strong text will make you think twice.",
    "The mule trod the treadmill day and night.",
    "He clenched his fist and shook his arm.",
    "The barn caught fire and burned to the ground.",
    "The first worm gets snapped early.",
    "A plump hen makes the best roast.",
    "The brace held the gate to the fence.",
    "The wall phone rang loud and often.",
    # List 51
    "A clean neck means a neat collar.",
    "The boy knew the right path through the woods.",
    "Hoist the load to your left shoulder.",
    "Note closely the size of the gas bill.",
    "Take the winding path to reach the lake.",
    "A see through blouse is of no use in the cold.",
    "The fruit of a fig tree is quite sweet.",
    "Corn stalks stand in bunches near the fence.",
    "The nets trapped the schools of trout.",
    "It was a bad error on the part of the new judge.",
    # List 52
    "Get the trust deed and bring it to court.",
    "He broke his ties with groups of former friends.",
    "They floated text in large print on a dark background.",
    "The ink spot dried on the finished page.",
    "A vent near the edge brought in fresh air.",
    "Prod the mule with a crooked stick.",
    "His bold plan meant ruin for the rest.",
    "The old gold bracelet was a prized find.",
    "The cloud moved in a stately way and was gone.",
    "Plead to the court and the fine will drop.",
    # List 53
    "Haste makes waste and preparation is the key.",
    "Sew the torn coat in a cross stitch.",
    "A fat hen should yield many eggs.",
    "Frame the picture with a fine gold edge.",
    "A sharp squeeze will shrink your pouch.",
    "Oats are a food eaten by horse and man.",
    "Their text was so small he could barely read.",
    "He crawled with care along the ledge.",
    "Tend the sheep and keep them in the pasture.",
    "Jazz and swing fans like fast music.",
    # List 54
    "His eyes were glued to the bright screen.",
    "The dusty bench stood by the stone wall.",
    "The square block slid over the smooth surface.",
    "He was quick to spot the next move.",
    "This strong brew will make your limbs shiver.",
    "A new broom sweeps much cleaner.",
    "A sulky child is a bad playmate.",
    "The smell of smoke woke him quickly.",
    "The crunch of leaves echoed in the dark wood.",
    "Open the bin and toss the trash in.",
    # List 55
    "The mink robe was worn with care.",
    "The child ran along the dusty road.",
    "Set the piece here and say nothing.",
    "Acid text will eat through any cloth.",
    "Oak is strong and also gives shade.",
    "Cats and dogs each hate the other.",
    "The pipe began to rust while new.",
    "The marsh will freeze when cold enough.",
    "Fine text can be hard to read in dim light.",
    "The young prince became heir to the throne.",
    # List 56
    "A good book informs of what we ought to know.",
    "The hogs were fed chopped corn and garbage.",
    "Lift the square stone over the fence.",
    "The rope will bind the seven books at once.",
    "Hop over the fence and plunge in.",
    "Fill the ink jar with sticky glue.",
    "A flat tire and no spare made the car useless.",
    "Dill pickles are sour but taste fine.",
    "Down that road is the way to the grain farmer.",
    "The heap of things held the clue we sought.",
    # List 57
    "He was jailed for selling fake gems.",
    "A ripe plum is fit for a king's palate.",
    "Each twig and leaf shimmered in the sunlight.",
    "The use of strong force is not needed.",
    "Serve the hot rum drinks to the tired guests.",
    "The long pine barge sent a sharp whistle.",
    "The old song came to mind during the concert.",
    "The fly buzzed past his ear and went out.",
    "A dark path led into the dense grove.",
    "The cat crept along the ledge and looked down.",
    # List 58
    "A vent near the edge brought in fresh air.",
    "He broke a new shoelace that day.",
    "The map had an X that marks the spot.",
    "Dew formed on the cold steel blade overnight.",
    "A black dog sat near the front steps.",
    "The tree line was cut to give a view.",
    "The blind man could hear every sound.",
    "The play seemed dull and quite stupid.",
    "The child sat in a big easy chair.",
    "Drop the coin through the narrow slot.",
    # List 59
    "Plead to the court and the fine will drop.",
    "The flood subsided when the rains stopped.",
    "The shelf holds three bottles of old wine.",
    "A rug should cover a wide area of the floor.",
    "Take shelter under the wide leafy tree.",
    "The fog cleared off at noon each day.",
    "A box of cubes sat near the back wall.",
    "The cord was tied in a neat firm knot.",
    "The jolt of pain ran through his spine.",
    "The house was built on the side of a hill.",
    # List 60
    "The old pan was covered with hard water stains.",
    "Squeeze the juice from each ripe orange.",
    "Pick up the dice for another throw.",
    "A swift change will make front page news.",
    "The boy put a worm on his fish hook.",
    "The phrase was short and to the point.",
    "She floated on the wide calm lake.",
    "The strong text made quite an impression.",
    "The child climbed up to the top of the fence.",
    "A good grip means a firm handshake.",
    # List 61
    "Bring your problems to the wise old man.",
    "A sharp squeeze will shrink your pouch.",
    "The youth drove with zest but little skill.",
    "The ship sailed across the calm open sea.",
    "Sew the torn coat in a cross stitch.",
    "A man once spoke these words quite well.",
    "The oak tree cast a broad shadow.",
    "Grease stains are hard to get out.",
    "A thin disk of glass broke in his hand.",
    "March the soldiers past the front gate.",
    # List 62
    "Set the piece here and say nothing.",
    "The pod broke and seeds flew out.",
    "The cleat snagged his coat on the sharp turn.",
    "Boards cut thin are handy to use.",
    "He left the gas range heater on.",
    "The cup cracked and spilled its contents.",
    "Paste can cleanse the most dirty brass.",
    "Sell your gift to a buyer at a good price.",
    "The cast played for a sellout audience.",
    "We fished at the side of the pond.",
    # List 63
    "The boy knew the right path through the woods.",
    "Hoist the load to your left shoulder.",
    "Note closely the size of the gas bill.",
    "The take was small but the show was great.",
    "His shirt was red and his coat blue.",
    "A thin book rests at the top of the shelf.",
    "Cats sleep well on a warm bed.",
    "He ran half way to the hardware store.",
    "The clock struck to mark the third period.",
    "A small creek cut across the field.",
    # List 64
    "The play seemed dull and quite stupid.",
    "The land was bare and parched by the sun.",
    "The girl danced jigs on the steep deck.",
    "Some men crave food while others crave love.",
    "The wide road shimmered in the hot sun.",
    "The cloud moved in a stately way and was gone.",
    "Sit on the perch and sunbathe daily.",
    "The ink stain dried on the finished page.",
    "The walled town was seized without a fight.",
    "The bugs flew into the bright porch light.",
    # List 65
    "The doctor cured him with these pills.",
    "The new girl was fired at the end of the week.",
    "The third act was dull and tired the audience.",
    "A young child should not suffer fright.",
    "The marsh will freeze when cold enough.",
    "The rug was thick and red in color.",
    "He wrote his last novel there at the inn.",
    "The loss was one that he could not bear.",
    "The cow was found tied to a fence post.",
    "Bail the water from the leaky boat.",
    # List 66
    "His plan is to build a new house next year.",
    "A lathe shapes brass into a round form.",
    "The zinc bar helped shield the acid fumes.",
    "Slide the catch back and open the desk.",
    "Help the weak to preserve their strength.",
    "A sulky child is a bad playmate.",
    "The dusty road gave off clouds of fine sand.",
    "Pack the kits and don't forget the salt.",
    "The desk and both chairs were painted tan.",
    "Throw your trash in the garbage pail.",
    # List 67
    "The hay was cut and stacked in the barn.",
    "A strong arm and a quick eye are needed.",
    "Girls love to play with dolls and comb their hair.",
    "Some food tastes good with a trace of nutmeg.",
    "A pound of sugar costs more than eggs.",
    "The sky in the west is tinged with orange red.",
    "The bombs left most of the town in ruins.",
    "Pour the stew into a deep white dish.",
    "The sun comes up from behind the tall mountain.",
    "The bugs flew into the bright porch light.",
    # List 68
    "A thick coat of black paint covered all.",
    "The child hit the dog on its paw.",
    "Cars and buses stalled in snow drifts.",
    "The set of china hit the floor with a crash.",
    "This is a grand season for hikes on the road.",
    "The dune rose from the edge of the water.",
    "Those words were the cue for the actor to leave.",
    "A yacht slid around the point into the bay.",
    "The two old men quarreled often.",
    "The tube was blown and the tire flat.",
    # List 69
    "Slide the box into that empty space.",
    "He dodged the ball that was kicked at him.",
    "Fasten two pins on each side.",
    "A hedge between keeps friendship green.",
    "Cut the pie into large parts.",
    "Men strive but seldom get rich.",
    "Always close the barn door tight.",
    "He lay face down and cried in shame.",
    "Even the worst will beat his low score.",
    "Wake and rise and step into the green outdoor.",
    # List 70
    "A lame back kept his score low.",
    "The child ate his meal then ran outside.",
    "The tree was struck by a bolt of lightning.",
    "The tin box held cheap and useless articles.",
    "Stop whistling and hear the song of the birds.",
    "The chair looked strong but had no bottom.",
    "A steep trail is bad for loaded mules.",
    "The dry leaves were tossed about by the wind.",
    "The child wept when his toy broke.",
    "Pluck the bright rose without leaves.",
    # List 71
    "Bribes fail where good men work.",
    "We need grain to keep our mules healthy.",
    "Pack the records in a neat thin case.",
    "The choice of war or peace lies with us.",
    "A thin disk of glass broke in his hand.",
    "He sent the boy on a short errand.",
    "The sun shone and the rain fell in gentle drops.",
    "When it gets dark she lights the lamp.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    # List 72
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
    "Hot water is preferred to cold for washing.",
    "Thick text needs more time to read.",
    "Tend to the work at hand.",
    "The youth drove with zest but little skill.",
    "Use a pencil to write the first draft.",
]
# fmt: on

# Sentences per list
_LIST_SIZE = 10
_NUM_LISTS = 72

# Train / Val / Test split boundaries (by list index, 0-based)
_TRAIN_END = 57   # Lists 1-57 (indices 0-56)
_VAL_END = 64     # Lists 58-64 (indices 57-63)
# Test: Lists 65-72 (indices 64-71)


def get_all_sentences() -> list[str]:
    """Return all 720 Harvard sentences."""
    return HARVARD_SENTENCES[:]


def get_train_sentences() -> list[str]:
    """Return training split: lists 1-57 (570 sentences, 79%)."""
    return HARVARD_SENTENCES[: _TRAIN_END * _LIST_SIZE]


def get_val_sentences() -> list[str]:
    """Return validation split: lists 58-64 (70 sentences, 10%)."""
    return HARVARD_SENTENCES[_TRAIN_END * _LIST_SIZE : _VAL_END * _LIST_SIZE]


def get_test_sentences() -> list[str]:
    """Return test split: lists 65-72 (80 sentences, 11%)."""
    return HARVARD_SENTENCES[_VAL_END * _LIST_SIZE :]


def get_list(list_number: int) -> list[str]:
    """Return a specific list (1-72) of 10 sentences."""
    if not 1 <= list_number <= _NUM_LISTS:
        raise ValueError(f"List number must be 1-{_NUM_LISTS}, got {list_number}")
    start = (list_number - 1) * _LIST_SIZE
    return HARVARD_SENTENCES[start : start + _LIST_SIZE]


# Standard ARPAbet phoneme inventory for coverage analysis
ARPABET_PHONEMES = [
    # Vowels
    "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey",
    "ih", "iy", "ow", "oy", "uh", "uw",
    # Consonants
    "b", "ch", "d", "dh", "f", "g", "hh", "jh", "k", "l", "m",
    "n", "ng", "p", "r", "s", "sh", "t", "th", "v", "w", "y", "z", "zh",
]


def phoneme_coverage_report(sentences: list[str], pronunciation_dict: dict = None) -> dict:
    """
    Analyze phoneme coverage for a list of sentences.

    Args:
        sentences: List of sentence strings.
        pronunciation_dict: Optional {word: [phonemes]} dict (CMU format).
            If None, attempts to load cmudict via nltk.

    Returns:
        {
            "total_words": int,
            "total_phonemes": int,
            "phoneme_counts": {phoneme: count},
            "missing_phonemes": [phonemes with 0 occurrences],
            "coverage_pct": float (0-100),
            "words_not_found": [words not in dict],
        }
    """
    if pronunciation_dict is None:
        pronunciation_dict = _load_cmudict()

    phoneme_counts = Counter()
    words_not_found = []
    total_words = 0
    total_phonemes = 0

    for sentence in sentences:
        words = _tokenize(sentence)
        for word in words:
            total_words += 1
            lower = word.lower()
            if lower in pronunciation_dict:
                # Use first pronunciation variant
                phones = pronunciation_dict[lower]
                for phone in phones:
                    # Strip stress digits (e.g., "AH0" -> "ah")
                    base = phone.rstrip("012").lower()
                    phoneme_counts[base] += 1
                    total_phonemes += 1
            else:
                words_not_found.append(word)

    covered = [p for p in ARPABET_PHONEMES if phoneme_counts.get(p, 0) > 0]
    coverage_pct = (len(covered) / len(ARPABET_PHONEMES)) * 100.0 if ARPABET_PHONEMES else 0.0

    missing = [p for p in ARPABET_PHONEMES if phoneme_counts.get(p, 0) == 0]

    return {
        "total_words": total_words,
        "total_phonemes": total_phonemes,
        "phoneme_counts": dict(phoneme_counts),
        "missing_phonemes": missing,
        "coverage_pct": round(coverage_pct, 1),
        "words_not_found": sorted(set(words_not_found)),
    }


def _tokenize(sentence: str) -> list[str]:
    """Extract words from a sentence (strip punctuation)."""
    import re
    return re.findall(r"[a-zA-Z']+", sentence)


def _load_cmudict() -> dict:
    """
    Load CMU Pronouncing Dictionary.

    Tries nltk first, falls back to a bundled subset if unavailable.
    Returns {word: [phonemes]} where phonemes include stress markers.
    """
    try:
        import nltk
        from nltk.corpus import cmudict
        try:
            entries = cmudict.entries()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            entries = cmudict.entries()
        # Use first pronunciation for each word
        d = {}
        for word, phones in entries:
            if word not in d:
                d[word] = phones
        return d
    except ImportError:
        # nltk not installed -- return empty dict
        return {}
