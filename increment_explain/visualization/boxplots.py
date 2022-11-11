import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from increment_explain.visualization.color import DEFAULT_COLOR_LIST

DEFAULT_COLOR_LIST = ['#1d4289']


def grouped_boxplot(data, feature_names, explainer_names, save_path, legend, width=0.5, min_space=0.1, feature_spacing=0.5, y_min=-1., y_max=1., legend_pos=0):
    n_explainer = len(explainer_names)
    n_features = len(feature_names)

    feature_names_short = [name[0:3]+'.' if len(name) > 3 else name[0:3] for name in feature_names]

    group_length = width * n_explainer + min_space * (n_explainer-1)
    x_range = np.array([x * group_length for x in range(n_features)])
    x_range = np.array([x_range[i] + feature_spacing * i for i in range(n_features)])
    x_locations = sorted(
        [feature_loc + explainer * (width + min_space) for explainer in range(n_explainer) for feature_loc in x_range])
    feature_x_locations = x_range + group_length / 2 - width / 2

    fig = plt.figure(0, figsize=[9, 4.8])
    ax = fig.add_subplot(111)
    x_position_index = 0
    for feature in feature_names:
        color_index = 0
        for explainer in explainer_names:
            if explainer == 'batch_total' or explainer == 'batch_interval':
                color = 'black'
            else:
                color = DEFAULT_COLOR_LIST[color_index]
                color_index += 1
            #ax.boxplot(
            #    data[explainer][feature], positions=[x_locations[x_position_index]],
            #    widths=width, patch_artist=True,
            #    medianprops={'color': 'red', 'linestyle': '-'},boxprops=dict(facecolor=color, color=None)
            #)
            violin_parts = ax.violinplot(
                data[explainer][feature], positions=[x_locations[x_position_index]],
                widths=width,
                showmeans=True, showmedians=True
            )
            for pc in violin_parts['bodies']:
                #pc.set_facecolor(color)
                #pc.set_edgecolor(color)
                pc.set_color(color)
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                vp = violin_parts[partname]
                vp.set_edgecolor(color)
                vp.set_linewidth(1)
            vp = violin_parts['cmedians']
            vp.set_edgecolor('red')
            vp.set_linewidth(1)

            x_position_index += 1
    x_position_index = 0
    color_index = 0
    for explainer in explainer_names:
        if explainer == 'batch_total' or explainer == 'batch_interval':
            color = 'black'
        else:
            color = DEFAULT_COLOR_LIST[color_index]
            color_index += 1
        legend_name = legend[x_position_index]
        ax.plot([], c=color, label=legend_name)
        x_position_index += 1

    ax.legend(edgecolor="0.8", fancybox=False, loc=legend_pos)
    ax.set_xticks(feature_x_locations, feature_names_short)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color=(0.5, 0.5, 0.5, 0.3), ls='--')
    ax.set_ylabel('SAGE Values')
    ax.set_xlabel('Features')
    plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":

    bat = [{'age': 0.028108668948204605, 'capital-gain': 0.05819510844543615, 'capital-loss': -0.009544486429741544, 'hours-per-week': 0.013155252465282052, 'fnlwgt': -0.0029857507201568488, 'workclass': -0.017613726611499247, 'education': 0.00042386958935468827, 'marital-status': 0.05439771678248133, 'occupation': 0.0016695371446097832, 'relationship': 0.08693410403102586, 'race': -0.005477809004474466, 'sex': 0.002669468345976513, 'native-country': -0.016564094138247252, 'education-num': 0.05385642606196482}, {'age': 0.027682255884510155, 'capital-gain': 0.055071822538957, 'capital-loss': -0.009050878850901029, 'hours-per-week': 0.01507412109689512, 'fnlwgt': -0.015753524269126337, 'workclass': -0.011765950551764718, 'education': -0.006686509514523585, 'marital-status': 0.08200240882771447, 'occupation': -0.003363941900762748, 'relationship': 0.06269993831493424, 'race': -0.007287603328323027, 'sex': 0.01845915407224611, 'native-country': -0.022423876746061333, 'education-num': 0.059211911293994245}, {'age': 0.02629933586076304, 'capital-gain': 0.047348888828515276, 'capital-loss': -0.008380769643279714, 'hours-per-week': 0.007628869030043197, 'fnlwgt': -0.009889991206343277, 'workclass': -0.0019752946425159617, 'education': -0.004365397345300091, 'marital-status': 0.08520484051636022, 'occupation': -0.0015253027093378555, 'relationship': 0.08201273651008828, 'race': -0.020244490079056546, 'sex': 0.0015202149566501612, 'native-country': -0.014535516949131752, 'education-num': 0.055141044503774064}, {'age': 0.02086520576527711, 'capital-gain': 0.05584322953650521, 'capital-loss': -0.004451512804291052, 'hours-per-week': 0.005407017217543419, 'fnlwgt': -0.014223539832536403, 'workclass': -0.010672812804902657, 'education': -0.0019274026641863853, 'marital-status': 0.07939688843223419, 'occupation': -0.00531566351257984, 'relationship': 0.08223375779888964, 'race': -0.012255298884018497, 'sex': 0.007240672284945291, 'native-country': -0.01716196642752966, 'education-num': 0.06409383725688973}, {'age': 0.03172187298173967, 'capital-gain': 0.05036333938639466, 'capital-loss': -0.017535865242490987, 'hours-per-week': 0.02263163451454883, 'fnlwgt': -0.010640433634705583, 'workclass': -0.016220548261482247, 'education': -0.006509064981150751, 'marital-status': 0.07753561376098746, 'occupation': -0.003088613293430229, 'relationship': 0.08879378828102573, 'race': -0.008228024623543877, 'sex': -0.001257477176132556, 'native-country': -0.016788444707527907, 'education-num': 0.05506402160315771}, {'age': 0.022885249213733968, 'capital-gain': 0.05188981662209141, 'capital-loss': -0.0027212247719945056, 'hours-per-week': 0.00735226411503803, 'fnlwgt': -0.030371798621983047, 'workclass': 0.0030405837219790945, 'education': -0.011568819869035936, 'marital-status': 0.05701380307681665, 'occupation': -0.006279217802575051, 'relationship': 0.10828757377516984, 'race': -0.014709792824596288, 'sex': 0.008093334216971913, 'native-country': -0.011103819544325651, 'education-num': 0.06916272816360407}, {'age': 0.017188197096873293, 'capital-gain': 0.055104508872217864, 'capital-loss': -0.0043284752591741665, 'hours-per-week': 0.006456527408449193, 'fnlwgt': -0.014094963296313321, 'workclass': -0.014463412048273606, 'education': -0.0016933164783028856, 'marital-status': 0.07471348400793593, 'occupation': -0.00790547489291557, 'relationship': 0.09988813376009567, 'race': -0.01540886370994617, 'sex': 0.00838854098765192, 'native-country': -0.017578997993545512, 'education-num': 0.056055099365500016}, {'age': 0.010354291915876435, 'capital-gain': 0.05330415567352984, 'capital-loss': -0.0018205928845302774, 'hours-per-week': 0.0092418954085724, 'fnlwgt': -0.016693608415608147, 'workclass': -0.012153118246136775, 'education': -0.0045776481276726, 'marital-status': 0.08940861686604011, 'occupation': -0.001520708784934167, 'relationship': 0.07302702595104182, 'race': -0.007689926107507798, 'sex': 0.009310812298269513, 'native-country': -0.020448972951357527, 'education-num': 0.06652569120600935}, {'age': 0.019755915593179133, 'capital-gain': 0.057325516840486486, 'capital-loss': -0.012006296604516168, 'hours-per-week': 0.011072765922383139, 'fnlwgt': -0.016524759309212735, 'workclass': -0.0012707924559163684, 'education': -0.0002908179329278292, 'marital-status': 0.0835673896045199, 'occupation': 0.005234255075447237, 'relationship': 0.0787406870013968, 'race': -0.008605934869527833, 'sex': 0.005992083815394923, 'native-country': -0.021159927171322278, 'education-num': 0.04913790222487249}, {'age': 0.02647903389514357, 'capital-gain': 0.05688205032555184, 'capital-loss': 0.0006749931554564515, 'hours-per-week': 0.012228737037665475, 'fnlwgt': -0.010864641096924343, 'workclass': -0.01321899165924379, 'education': 0.004665774153179211, 'marital-status': 0.05506181592877661, 'occupation': -0.005296510368831303, 'relationship': 0.09392146296134306, 'race': -0.009885204257821112, 'sex': -0.003951912535501598, 'native-country': -0.012028018331600254, 'education-num': 0.054405072119631005}]
    inc = [{'age': 0.026212565230670844, 'capital-gain': 0.051853815484481695, 'capital-loss': -0.03375947462819334, 'hours-per-week': -0.005952738049372549, 'fnlwgt': 0.010891015572607606, 'workclass': 0.007855645437528758, 'education': 0.020711098122370295, 'marital-status': 0.06818484889581049, 'occupation': 0.008780778089487512, 'relationship': 0.11479715867765075, 'race': -0.0396267917209024, 'sex': 0.01715652361833196, 'native-country': -0.01924195964756475, 'education-num': 0.05166071497440728}, {'age': 0.031953365210154086, 'capital-gain': 0.040233816057044174, 'capital-loss': 0.015308258533997407, 'hours-per-week': 0.024763094442210132, 'fnlwgt': -0.023864240785632842, 'workclass': -0.005790773616792286, 'education': 0.002009900888309228, 'marital-status': 0.09022362975547611, 'occupation': 0.018962049430597573, 'relationship': 0.06255732633063874, 'race': -0.029619480401851726, 'sex': 0.007644171018501113, 'native-country': -0.006982363425793644, 'education-num': 0.04791278779268208}, {'age': 0.03105109977374419, 'capital-gain': 0.050636700735616574, 'capital-loss': 0.0010303195419861496, 'hours-per-week': 0.006107991972910388, 'fnlwgt': -0.007916388612774319, 'workclass': 0.014082826220676083, 'education': -0.010737141118534802, 'marital-status': 0.06908178217647476, 'occupation': 0.031675961832971174, 'relationship': 0.059901137800125445, 'race': -0.007157037614547071, 'sex': -0.0009777636433992965, 'native-country': -0.007626602269783604, 'education-num': 0.058542561430969496}, {'age': 0.0019795192237329323, 'capital-gain': 0.06266937606807628, 'capital-loss': 0.014599549092842078, 'hours-per-week': 0.007318639276298247, 'fnlwgt': -0.010726128120754559, 'workclass': 0.004489314667250441, 'education': 0.0008595487221832177, 'marital-status': 0.0873553837252126, 'occupation': 0.005523453993294373, 'relationship': 0.07941329917704586, 'race': -0.016871830093080787, 'sex': 0.0033822590452917494, 'native-country': -0.024440122334842395, 'education-num': 0.08584025865593141}, {'age': 0.011240568934383448, 'capital-gain': 0.04916588978528996, 'capital-loss': -0.002950679370124982, 'hours-per-week': 0.031098753850121487, 'fnlwgt': -0.01817694217174261, 'workclass': 0.0011091359097070786, 'education': 0.005897980198222674, 'marital-status': 0.08630570758662444, 'occupation': -0.010636354979754884, 'relationship': 0.10562996823761453, 'race': -0.03278296714088403, 'sex': 0.01516986822424592, 'native-country': 0.00030476243689187574, 'education-num': 0.04474164286708921}, {'age': 0.02527621561549269, 'capital-gain': 0.05942634621416649, 'capital-loss': -0.00778843166475, 'hours-per-week': -0.006697969601842125, 'fnlwgt': -0.002620768074243089, 'workclass': -0.011759245580332561, 'education': 0.009436931521865479, 'marital-status': 0.050490512312519775, 'occupation': -0.007253971655294637, 'relationship': 0.1055138636402115, 'race': -0.004324591297986123, 'sex': 0.00974487665515802, 'native-country': -0.016807565593019727, 'education-num': 0.08101708872067219}, {'age': 0.013665420626564943, 'capital-gain': 0.04084367707389386, 'capital-loss': -0.010685161970382525, 'hours-per-week': 0.026638396909574487, 'fnlwgt': -0.006610950459774061, 'workclass': -0.01297298975404451, 'education': 0.014663972299887171, 'marital-status': 0.05502675604932766, 'occupation': -0.01721292825385452, 'relationship': 0.10073932601856056, 'race': 0.01145610023483624, 'sex': -0.006986945163094203, 'native-country': -0.0017781354711467162, 'education-num': 0.06348274995816154}, {'age': -0.009320204798651387, 'capital-gain': 0.06737366771417695, 'capital-loss': 0.026711433068659206, 'hours-per-week': 0.01883960509326243, 'fnlwgt': 0.007472164747561111, 'workclass': -0.047188634961137156, 'education': -0.003399483946126723, 'marital-status': 0.10785094801889761, 'occupation': -0.01781491820830615, 'relationship': 0.06674819763646075, 'race': -0.0032157086034249644, 'sex': 0.00456514963644816, 'native-country': -0.010325849140494201, 'education-num': 0.07132928098600357}, {'age': 0.024523050501542555, 'capital-gain': 0.07033876936681563, 'capital-loss': 0.002557534743437581, 'hours-per-week': 0.02273157756914485, 'fnlwgt': -0.01480068897641932, 'workclass': -0.017999555202484696, 'education': 0.011004783289738721, 'marital-status': 0.09243051384466423, 'occupation': 0.01293094691824848, 'relationship': 0.05214557258523368, 'race': 0.004067037722099184, 'sex': -0.007338460285802911, 'native-country': -0.04306589901901369, 'education-num': 0.07354986167466582}, {'age': 0.027940164264492553, 'capital-gain': 0.03407085118913663, 'capital-loss': 0.0038879367482070283, 'hours-per-week': 0.0037577188617372538, 'fnlwgt': -0.0018272174107804018, 'workclass': 0.008329442815579695, 'education': -0.011475516553836418, 'marital-status': 0.06979644873463138, 'occupation': 0.03565973822196416, 'relationship': 0.0738045663238286, 'race': -0.012000996879253012, 'sex': 0.001722160244286283, 'native-country': -0.017234569261312706, 'education-num': 0.05323684425911136}]

    feature_names = list(inc[0].keys())

    bat = pd.DataFrame(bat)
    inc = pd.DataFrame(inc)



    data = {'batch_total': bat, 'incremental': inc}

    explainer_names = list(data.keys())

    grouped_boxplot(
        data,
        feature_names,
        explainer_names,
        save_path=None,
        legend=['Batch SAGE (left)', 'iSAGE (right)'], legend_pos=2,
        width=0.5, min_space=0.1, feature_spacing=0.5,
        y_min=-0.06, y_max=0.12
    )